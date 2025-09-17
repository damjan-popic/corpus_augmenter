#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_classla_shards.py
---------------------
End-to-end, resilient Classla augmentation for HUGE CoNLL-U files:
  1) Split input file(s) into shards by sentence count
  2) Augment each shard (pretokenized; only fill missing layers)
  3) Write incrementally with checkpoints (resume-safe)
  4) Join augmented shards into the final output file(s)

Features
- Pretokenized, no-ssplit pipeline → exact token alignment
- Never overwrites existing annotations; only fills gaps
- NER comes from TOKENS (token.ner), written to MISC (UD) or last column (JOS); preserves existing MISC keys
- SpaceAfter=No preserved
- Rich progress bars and periodic flush/fsync
- Resume with --resume after any interruption
- Optional per-processor batch “hints” collapse to a safe global sent-batch
"""

from __future__ import annotations
import argparse, os, sys, json, time, errno
from pathlib import Path
from typing import List, Dict, Iterable, Tuple, Optional

# Third-party
import classla

# Pretty terminal
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn
from rich.theme import Theme

ENC = "utf-8"
CONL_EXTS = {".conllu", ".conll"}

console = Console(theme=Theme({
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "info": "cyan",
    "hint": "bold blue"
}))

# =========================
# Small utilities
# =========================

def list_sources(inp: Path) -> List[Path]:
    if inp.is_file() and inp.suffix.lower() in CONL_EXTS:
        return [inp]
    elif inp.is_dir():
        return sorted([p for p in inp.rglob("*") if p.suffix.lower() in CONL_EXTS])
    else:
        console.print(f"[err] Input path not found or not a .conllu/.conll file/dir: {inp}")
        sys.exit(1)

def iter_sentence_blocks(path: Path) -> Iterable[List[str]]:
    """Yield sentence blocks (list of lines including token lines and (optional) sent-level comments).
    Splits on blank lines. Leaves leading newdoc/comments with the sentence that follows."""
    buf = []
    seen_tok = False
    with path.open(encoding=ENC) as f:
        for ln in f:
            if ln.strip():
                buf.append(ln)
                if not ln.startswith("#"):
                    seen_tok = True
            else:
                if seen_tok:
                    yield buf
                    buf = []
                    seen_tok = False
                else:
                    if buf:
                        buf.append("\n")
        if seen_tok and buf:
            yield buf

def count_sentences(path: Path) -> int:
    n = 0
    for _ in iter_sentence_blocks(path):
        n += 1
    return n

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ckpt_paths(out_path: Path, checkpoint_dir: Optional[Path]) -> Tuple[Path, Path]:
    ckdir = checkpoint_dir or out_path.parent
    safe_mkdir(ckdir)
    ckpt = ckdir / (out_path.name + ".ckpt")
    part = out_path.with_suffix(out_path.suffix + ".part")
    return ckpt, part

def load_ckpt(ckpt: Path) -> int:
    try:
        return int(json.loads(ckpt.read_text(encoding=ENC)).get("done", 0))
    except Exception:
        return 0

def save_ckpt(ckpt: Path, done: int):
    tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
    tmp.write_text(json.dumps({"done": int(done)}), encoding=ENC)
    tmp.replace(ckpt)

def safe_replace(src: Path, dst: Path, retries: int = 3, sleep: float = 0.2):
    """Robust replace to handle transient locks on /mnt/c."""
    for i in range(retries):
        try:
            src.replace(dst)
            return
        except OSError as e:
            if i == retries - 1 or e.errno not in (errno.EACCES, errno.EBUSY):
                raise
            time.sleep(sleep)

def parse_batches_arg(s: Optional[str], default_sent_batch: int) -> int:
    """
    Accept a string like "tokenize=30000,pos=6000,lemma=15000,depparse=300,ner=800"
    and convert to an effective global sent-batch for safety:
    effective = min(default_sent_batch, any provided numbers).
    """
    if not s:
        return default_sent_batch
    try:
        pairs = {}
        for part in s.split(","):
            if not part.strip():
                continue
            k, v = part.split("=")
            pairs[k.strip()] = int(v.strip())
        effective = min([default_sent_batch] + list(pairs.values()))
        if effective < default_sent_batch:
            console.print(f"[hint] Using conservative batch size {effective} derived from --batches (was {default_sent_batch}).")
        return max(1, effective)
    except Exception:
        console.print("[warn] Could not parse --batches; ignoring and using --sent-batch as-is.")
        return default_sent_batch

# =========================
# CoNLL-U overlay helpers
# =========================

def detect_format(cols: List[str]) -> str:
    """Rough schema detection."""
    if len(cols) >= 10:
        return "ud"   # UPOS,XPOS,FEATS,HEAD,DEPREL,MISC
    elif len(cols) >= 7:
        return "jos"  # 7+ columns with JOS-style
    return "unknown"

def needs_layers(cols: List[str], fmt: str) -> Dict[str, bool]:
    needs = {"lemma": False, "pos": False, "feats": False, "deprel": False, "ner": False}
    # id form lemma upos xpos feats head deprel deps misc
    if len(cols) < 3 or cols[2] in ("", "_"):
        needs["lemma"] = True
    if len(cols) < 5 or cols[3] in ("", "_") or cols[4] in ("", "_"):
        needs["pos"] = True
    if len(cols) < 6 or cols[5] in ("", "_"):
        needs["feats"] = True
    if len(cols) < 8 or cols[6] in ("", "_") or cols[7] in ("", "_"):
        needs["deprel"] = True
    # NER
    if fmt == "ud":
        misc = cols[9] if len(cols) > 9 else ""
        if (not misc) or ("NER=" not in misc) or misc.endswith("NER=O"):
            needs["ner"] = True
    elif fmt == "jos":
        last = cols[-1] if cols else ""
        if last in ("", "_", "O", "NER=O") or "NER=" not in last:
            needs["ner"] = True
    else:
        needs["ner"] = True
    return needs

def add_misc_kv(misc: str, key: str, value: str) -> str:
    if not misc or misc == "_":
        return f"{key}={value}"
    parts = [p for p in misc.split("|") if p and not p.startswith(f"{key}=")]
    parts.append(f"{key}={value}")
    return "|".join(parts)

def overlay(cols: List[str], w, fmt: str, needs: Dict[str, bool], ner_tag: Optional[str] = None) -> List[str]:
    """Overlay missing layers using classla word + (optionally) token NER."""
    # lemma
    if needs.get("lemma") and len(cols) > 2 and (cols[2] in ("_", "")) and getattr(w, "lemma", None):
        cols[2] = w.lemma
    # upos/xpos
    if needs.get("pos"):
        if len(cols) > 3 and cols[3] in ("_", "") and getattr(w, "upos", None):
            cols[3] = w.upos
        if len(cols) > 4 and cols[4] in ("_", "") and getattr(w, "xpos", None):
            cols[4] = w.xpos
    # feats
    if needs.get("feats") and len(cols) > 5 and cols[5] in ("_", "") and getattr(w, "feats", None):
        cols[5] = w.feats
    # head / deprel
    if needs.get("deprel"):
        if len(cols) > 6 and cols[6] in ("_", "") and getattr(w, "head", None) is not None:
            cols[6] = str(w.head)
        if len(cols) > 7 and cols[7] in ("_", "") and getattr(w, "deprel", None):
            cols[7] = w.deprel
    # ner (token-level)
    if ner_tag and needs.get("ner"):
        if fmt == "ud":
            if len(cols) > 9:
                cols[9] = add_misc_kv(cols[9], "NER", ner_tag)
            else:
                while len(cols) < 10:
                    cols.append("_")
                cols[9] = add_misc_kv(cols[9], "NER", ner_tag)
        elif fmt == "jos":
            if cols[-1] in ("", "_", "O", "NER=O") or "NER=" not in cols[-1]:
                cols[-1] = f"NER={ner_tag}"
    return cols

def sentence_to_forms(sent_lines: List[str]) -> List[str]:
    forms = []
    for ln in sent_lines:
        if ln.startswith("#") or not ln.strip():
            continue
        cols = ln.rstrip("\n").split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        forms.append(cols[1])
    return forms

def apply_updates(sent_lines: List[str], doc_sent) -> List[str]:
    """
    Return updated sentence lines with overlays applied.
    Assumes pretokenized alignment: len(words) == len(tokens), and
    MWT/empty nodes are skipped for alignment.
    """
    updated = []
    words_iter = iter(doc_sent.words)
    tokens_iter = iter(doc_sent.tokens)

    for ln in sent_lines:
        if ln.startswith("#") or not ln.strip():
            updated.append(ln)
            continue
        cols = ln.rstrip("\n").split("\t")

        # Keep MWT/empty nodes unchanged
        if "-" in cols[0] or "." in cols[0]:
            updated.append(ln)
            continue

        fmt = detect_format(cols)
        needs = needs_layers(cols, fmt)
        try:
            w = next(words_iter)
            tok = next(tokens_iter)
        except StopIteration:
            # alignment issue; keep original
            updated.append(ln)
            continue

        ner_tag = getattr(tok, "ner", None)  # <-- token-level NER
        new_cols = overlay(cols, w, fmt, needs, ner_tag=ner_tag)
        updated.append("\t".join(new_cols) + "\n")
    return updated

# =========================
# Classla pipeline
# =========================

def build_pipeline(lang: str, use_gpu: bool) -> classla.Pipeline:
    return classla.Pipeline(
        lang=lang,
        processors="tokenize,pos,lemma,depparse,ner",
        use_gpu=use_gpu,
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
    )

def augment_batch(batch: List[List[str]], nlp: classla.Pipeline) -> List[List[str]]:
    """Run one forward pass on a batch of sentence blocks."""
    inputs = [sentence_to_forms(s) for s in batch]
    doc = nlp(inputs)
    out = []
    for sent_lines, doc_sent in zip(batch, doc.sentences):
        out.append(apply_updates(sent_lines, doc_sent))
    return out

# =========================
# Sharding
# =========================

def shard_one_file(src: Path, shard_dir: Path, shard_sents: int) -> Tuple[List[Path], int]:
    """Split src into shards of shard_sents sentences. Returns (shard_paths, total_sents)."""
    safe_mkdir(shard_dir)
    shard_paths = []
    sent_count = 0
    shard_idx = 0
    w = None

    def open_new():
        nonlocal shard_idx, w
        shard_idx += 1
        p = shard_dir / f"{src.stem}.part{shard_idx:04d}.conllu"
        shard_paths.append(p)
        w = p.open("w", encoding=ENC)

    open_new()
    for block in iter_sentence_blocks(src):
        if sent_count > 0 and sent_count % shard_sents == 0:
            w.close()
            open_new()
        # write the sentence & a blank line
        w.writelines(block)
        if block and block[-1].strip():
            w.write("\n")
        sent_count += 1
    if w:
        w.close()
    return shard_paths, sent_count

def concat_augmented_shards(aug_dir: Path, pattern_prefix: str, final_out: Path):
    """Concatenate augmented shards (sorted) into final_out."""
    part_list = sorted(aug_dir.glob(f"{pattern_prefix}.part*.aug.conllu"))
    if not part_list:
        console.print(f"[warn] No augmented shards found for {pattern_prefix} in {aug_dir}")
        return
    with final_out.open("w", encoding=ENC) as out:
        for p in part_list:
            with p.open(encoding=ENC) as f:
                for ln in f:
                    out.write(ln)
    console.print(f"[ok] Joined {len(part_list)} shards → {final_out.name}")

# =========================
# Augment loop (incremental)
# =========================

def write_leading_comments_if_any(src_path: Path, w):
    with src_path.open(encoding=ENC) as f:
        for ln in f:
            if ln.startswith("#"):
                w.write(ln)
            elif ln.strip() == "":
                w.write("\n")
                break
            else:
                break

def process_shard(shard_path: Path, out_dir: Path, nlp: classla.Pipeline,
                  sent_batch: int, write_every: int, checkpoint_dir: Optional[Path],
                  resume: bool, progress: Progress, task_id) -> bool:
    """
    Augment one shard into out_dir/<shard>.aug.conllu, with .part and .ckpt for resume.
    Returns True on success.
    """
    safe_mkdir(out_dir)
    out_path = out_dir / (shard_path.stem + ".aug.conllu")
    ckpt, part = ckpt_paths(out_path, checkpoint_dir)
    done = load_ckpt(ckpt) if resume else 0

    # Load all sentences in this shard (iterating twice is ok for shards)
    sents = list(iter_sentence_blocks(shard_path))
    total = len(sents)
    if done >= total:
        if part.exists():
            safe_replace(part, out_path)
        progress.update(task_id, completed=total)
        return True

    mode = "a" if (resume and part.exists()) else "w"
    with part.open(mode, encoding=ENC) as w:
        # If new write, also copy any leading comments (rare in shards)
        if mode == "w":
            write_leading_comments_if_any(shard_path, w)

        batch = []
        idx = done

        # fast-forward progress bar to 'done'
        progress.update(task_id, completed=done)

        for s_i in range(done, total):
            batch.append(sents[s_i])
            # dispatch when we hit batch size or last sentence
            if len(batch) >= sent_batch or s_i == total - 1:
                deltas = augment_batch(batch, nlp)
                for upd in deltas:
                    w.writelines(upd)
                    if upd and upd[-1].strip():
                        w.write("\n")
                    idx += 1
                save_ckpt(ckpt, idx)

                # periodic flush/fsync
                # flush every N batches (by index of completed batches)
                if write_every <= 1 or ((idx // sent_batch) % write_every == 0):
                    w.flush()
                    os.fsync(w.fileno())

                # progress
                progress.update(task_id, advance=len(deltas))
                batch.clear()

    # finalize
    safe_replace(part, out_path)
    if ckpt.exists():
        ckpt.unlink(missing_ok=True)
    return True

# =========================
# Stats
# =========================

def sentence_token_count(sent_lines: List[str]) -> int:
    c = 0
    for ln in sent_lines:
        if ln.startswith("#") or not ln.strip():
            continue
        cols = ln.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        c += 1
    return c

def write_stats(stats_dir: Path, src_file: Path, final_out: Path, total_sents: int):
    safe_mkdir(stats_dir)
    # quick token recount
    toks = 0
    for s in iter_sentence_blocks(final_out):
        toks += sentence_token_count(s)
    rep = {
        "source": str(src_file),
        "output": str(final_out),
        "sentences": total_sents,
        "tokens": toks,
        "timestamp": int(time.time())
    }
    (stats_dir / f"{final_out.stem}.stats.json").write_text(json.dumps(rep, indent=2), encoding=ENC)

# =========================
# Main orchestration
# =========================

def build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        transient=False,
        console=console
    )

def main():
    ap = argparse.ArgumentParser(description="Shard → augment → join Classla pipeline (resilient, resumable).")
    ap.add_argument("--mode", choices=["augment"], required=True)
    ap.add_argument("--lang", default="sl", help="Language code for Classla models (default: sl).")
    ap.add_argument("--in", dest="inp", required=True, help="Input .conllu file OR directory of .conllu files.")
    ap.add_argument("--out", dest="out", required=True, help="Output directory for final joined files.")
    ap.add_argument("--stats", dest="stats", required=True, help="Directory to write stats JSON.")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    ap.add_argument("--shard-sents", type=int, default=1_000_000, help="Sentences per shard (default: 1,000,000).")
    ap.add_argument("--sent-batch", type=int, default=128, help="Sentences per forward pass (default: 128).")
    ap.add_argument("--batches", default=None, help="Comma list: tokenize=..,pos=..,lemma=..,depparse=..,ner=.. (collapsed to safe global).")
    ap.add_argument("--resume", action="store_true", help="Resume from existing .ckpt/.part files.")
    ap.add_argument("--checkpoint-dir", default=None, help="Directory to store checkpoints (default: alongside shard output).")
    ap.add_argument("--write-every", type=int, default=2, help="Flush/fsync every N batches (default: 2).")
    ap.add_argument("--only", default=None, help="Process only files whose basename matches this (optional).")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out); safe_mkdir(out_dir)
    stats_dir = Path(args.stats); safe_mkdir(stats_dir)
    prefer_gpu = not args.no_gpu

    sources = list_sources(inp)
    if args.only:
        sources = [p for p in sources if p.name == args.only or p.stem == args.only]
        if not sources:
            console.print(f"[err] --only {args.only} did not match any input file(s).")
            sys.exit(1)

    # collapse per-processor batches to a safe global batch
    effective_batch = parse_batches_arg(args.batches, args.sent_batch)

    # Build one pipeline per run (reused across shards/files)
    console.print("[info] Loading Classla pipeline…")
    nlp = build_pipeline(args.lang, use_gpu=prefer_gpu)
    console.print(f"[ok] Augment pipeline ready → {[k for k in nlp.processors.keys()]}")

    # Top-level progress
    with build_progress() as progress:
        for src in sources:
            base = src.stem
            console.print(f"[info] Sharding: {src.name}")
            shards_root = out_dir / "_shards" / base
            aug_shards_root = out_dir / "_aug_shards" / base
            safe_mkdir(shards_root)
            safe_mkdir(aug_shards_root)

            # If already sharded, reuse; else create shards
            shard_paths = sorted(shards_root.glob(f"{base}.part*.conllu"))
            total_sents = None
            if not shard_paths:
                shard_paths, total_sents = shard_one_file(src, shards_root, args.shard_sents)
                console.print(f"[ok] Created {len(shard_paths)} shard(s) for {src.name}")
            else:
                # estimate sentence count across shards (fast)
                total_sents = 0
                for p in shard_paths:
                    for _ in iter_sentence_blocks(p):
                        total_sents += 1

            # Progress task per shard
            for shard in shard_paths:
                # Count sentences in shard (fast)
                shard_sents = 0
                for _ in iter_sentence_blocks(shard):
                    shard_sents += 1
                t = progress.add_task(f"[info] Augment {shard.name}", total=shard_sents)

                ok = process_shard(
                    shard_path=shard,
                    out_dir=aug_shards_root,
                    nlp=nlp,
                    sent_batch=effective_batch,
                    write_every=args.write_every,
                    checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
                    resume=args.resume,
                    progress=progress,
                    task_id=t
                )
                if not ok:
                    console.print(f"[err] Failed on shard {shard.name}; aborting file {src.name}")
                    sys.exit(1)

            # Join augmented shards → final
            final_out = out_dir / f"{base}.aug.conllu"
            concat_augmented_shards(aug_shards_root, base, final_out)

            # Stats
            if total_sents is None:
                total_sents = count_sentences(src)
            write_stats(stats_dir, src, final_out, total_sents)
            console.print(f"[ok] Finished {src.name} → {final_out.name}")

if __name__ == "__main__":
    # Stability knobs that help in WSL
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    try:
        # best-effort; harmless if it fails
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 4096:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    except Exception:
        pass
    main()
