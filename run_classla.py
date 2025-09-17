#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classla all-processors pipeline (GPU-ready) with robust AUGMENT + COMBINE.

Features
--------
- augment: fills ONLY missing layers in CoNLL-U/JOS/7-col files
  * Uses pretokenized mode (no re-tokenization), no sentence split
  * Overlays lemma/upos/xpos/feats/head/deprel + NER (in MISC for UD; last col for JOS)
  * Preserves existing annotations (never overwrites non-missing values)
  * Preserves existing NER if present and not 'O'

- combine: merges all *.conllu|*.conll recursively from --in into one file

Performance knobs
-----------------
--sent-batch N            → sentence microbatch (outer), e.g. 512/768
--batches k=v,...         → per-processor batch sizes (inner), e.g. depparse=256,ner=512,pos=1500,lemma=1500

Best practice (RTX 2000 16GB)
------------------------------
Tweets:              --sent-batch 768 --batches depparse=256,ner=512,pos=2000,lemma=2000
Mixed/long forums:   --sent-batch 512 --batches depparse=128,ner=256,pos=1500,lemma=1500
OOM-resistant:       --sent-batch 256 --batches depparse=128,ner=256,pos=1000,lemma=1000
You can also set:  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CLI examples
------------
python -u run_classla.py --mode augment --lang sl \
  --in datasets/conllu_in --out out_augmented --stats out_stats \
  --sent-batch 512 --batches depparse=256,ner=512,pos=1500,lemma=1500

python -u run_classla.py --mode combine \
  --in datasets/conllu_in --out out_combined --stats out_stats \
  --combined-name combined.conllu
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import io
import json
import shutil
from typing import List, Dict, Tuple, Optional

# Third-party
import classla

from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn
from rich.theme import Theme


# -----------------------
# Console / globals
# -----------------------
THEME = Theme({
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "info": "cyan",
    "head": "bold",
})

console = Console(theme=THEME, force_terminal=True, highlight=False)

ENC = "utf-8"
CONL_EXTS = {".conllu", ".conll"}

# -----------------------
# Argparse
# -----------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Classla all-processors pipeline (augment/combine) with GPU + per-processor batches."
    )
    ap.add_argument("--mode", choices=["augment", "combine"], required=True,
                    help="augment: fill missing layers in CoNLL-U; combine: merge files.")
    ap.add_argument("--in", dest="inp", required=True, help="Input directory (recursively scanned).")
    ap.add_argument("--out", dest="out", required=True, help="Output directory.")
    ap.add_argument("--stats", dest="stats", required=True, help="Directory to write simple stats JSON/TSV.")
    ap.add_argument("--lang", default="sl", help="Language code for Classla models (default: sl).")

    # Performance knobs
    ap.add_argument("--sent-batch", type=int, default=512,
                    help="Sentence microbatch size per forward pass (outer batching).")
    ap.add_argument("--batches",
                    help="Per-processor batch sizes, e.g. 'depparse=256,ner=512,pos=1500,lemma=1500'.")

    ap.add_argument("--combined-name", default="combined.conllu",
                    help="[combine] Output filename (default: combined.conllu).")

    ap.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    return ap


# -----------------------
# Utils
# -----------------------
def parse_batches(bstr: Optional[str]) -> Dict[str, int]:
    if not bstr:
        return {}
    out: Dict[str, int] = {}
    for kv in bstr.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Bad --batches item (missing '='): {kv}")
        k, v = kv.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def list_conllu_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in CONL_EXTS]


def atomic_write_text(path: Path, text: str, encoding: str = ENC):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def detect_format(cols: List[str]) -> str:
    """
    Return 'ud' for 10+ columns (CoNLL-U), 'jos' for >=7 (reduced/JOS), else 'unknown'.
    """
    if len(cols) >= 10:
        return "ud"
    if len(cols) >= 7:
        return "jos"
    return "unknown"


# -----------------------
# CoNLL-U parsing
# -----------------------
def iter_sentences_conllu(lines: List[str]):
    """
    Yield tuples: (comment_lines[list[str]], token_lines[list[str]])
    Separates sentences on blank lines.
    """
    comments: List[str] = []
    toks: List[str] = []
    for ln in lines:
        if not ln.strip():
            if comments or toks:
                yield comments, toks
            comments, toks = [], []
            continue
        if ln.startswith("#"):
            comments.append(ln)
        else:
            toks.append(ln)
    # last
    if comments or toks:
        yield comments, toks


def split_cols(row: str) -> List[str]:
    return row.split("\t")


def join_cols(cols: List[str]) -> str:
    return "\t".join(cols)


# -----------------------
# Missing-layer detection
# -----------------------
def needs_layers_for_sentence(token_rows: List[str], fmt: str) -> Dict[str, bool]:
    """
    Decide which layers are missing in a sentence.
    Returns flags for: lemma, pos, feats, deprel, ner
    """
    flags = {"lemma": False, "pos": False, "feats": False, "deprel": False, "ner": False}
    for row in token_rows:
        if "-" in row.split("\t", 2)[0] or "." in row.split("\t", 2)[0]:
            # skip MWEs or empty nodes
            continue
        cols = split_cols(row)
        # lemma
        if len(cols) < 3 or cols[2] in ("_", ""):
            flags["lemma"] = True
        # pos (UPOS/XPOS)
        if len(cols) < 5 or cols[3] in ("_", "") or cols[4] in ("_", ""):
            flags["pos"] = True
        # feats
        if len(cols) < 6 or cols[5] in ("_", ""):
            flags["feats"] = True
        # head/deprel
        if len(cols) < 8 or cols[6] in ("_", "") or cols[7] in ("_", ""):
            flags["deprel"] = True
        # ner
        if fmt == "ud":
            misc = cols[9] if len(cols) >= 10 else ""
            if (not misc) or ("NER=" not in misc) or misc.endswith("NER=O"):
                flags["ner"] = True
        elif fmt == "jos":
            last = cols[-1] if cols else ""
            if (not last) or last in ("_", "O", "NER=O"):
                flags["ner"] = True
    return flags


# -----------------------
# Overlay logic
# -----------------------
def overlay_token(cols: List[str], w, fmt: str, needs: Dict[str, bool]) -> List[str]:
    """Overlay missing fields with model outputs from Classla token `w`."""
    # lemma
    if needs.get("lemma") and len(cols) > 2 and (cols[2] in ("_", "")) and getattr(w, "lemma", None):
        cols[2] = w.lemma
    # pos
    if needs.get("pos"):
        if len(cols) > 3 and cols[3] in ("_", "") and getattr(w, "upos", None):
            cols[3] = w.upos
        if len(cols) > 4 and cols[4] in ("_", "") and getattr(w, "xpos", None):
            cols[4] = w.xpos
    # feats
    if needs.get("feats") and len(cols) > 5 and cols[5] in ("_", "") and getattr(w, "feats", None):
        cols[5] = w.feats
    # deps
    if needs.get("deprel"):
        if len(cols) > 6 and cols[6] in ("_", "") and getattr(w, "head", None) is not None:
            cols[6] = str(w.head)
        if len(cols) > 7 and cols[7] in ("_", "") and getattr(w, "deprel", None):
            cols[7] = w.deprel
    # ner
    ner_tag = getattr(w, "ner", None)
    if needs.get("ner") and ner_tag:
        if fmt == "ud":
            if len(cols) < 10:
                cols += ["_"] * (10 - len(cols))
            misc = cols[9]
            # preserve existing NER if not O
            if misc and "NER=" in misc and not misc.endswith("NER=O"):
                pass
            else:
                cols[9] = (misc + "|" if misc and misc != "_" else "") + f"NER={ner_tag}"
        elif fmt == "jos":
            # last column holds NER in our reduced format
            last = cols[-1] if cols else ""
            if last and last not in ("_", "O", "NER=O"):
                pass
            else:
                if cols:
                    cols[-1] = f"NER={ner_tag}"
                else:
                    cols.append(f"NER={ner_tag}")
    return cols


# -----------------------
# Pipeline factory
# -----------------------
def build_pipeline(lang: str, prefer_gpu: bool, batches: Dict[str, int]) -> classla.Pipeline:
    pipe_kwargs = dict(
        lang=lang,
        processors="tokenize,pos,lemma,depparse,ner",
        use_gpu=prefer_gpu,
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
    )
    # per-processor batch sizes (ignored if unsupported)
    if "pos" in batches:
        pipe_kwargs["pos_batch_size"] = batches["pos"]
    if "lemma" in batches:
        pipe_kwargs["lemma_batch_size"] = batches["lemma"]
    if "depparse" in batches:
        pipe_kwargs["depparse_batch_size"] = batches["depparse"]
    if "ner" in batches:
        pipe_kwargs["ner_batch_size"] = batches["ner"]
    if "tokenize" in batches:
        pipe_kwargs["tokenize_batch_size"] = batches["tokenize"]

    nlp = classla.Pipeline(**pipe_kwargs)
    console.print(f"Augment pipeline ready → { {k: type(v) for k,v in nlp.processors.items()} }", style="info")
    return nlp


# -----------------------
# AUGMENT core (batch mode)
# -----------------------
def augment_file(
    in_path: Path,
    out_path: Path,
    nlp: classla.Pipeline,
    sent_batch: int,
    progress: Progress,
    p_task: int,
    total_sents_task: Optional[int] = None,
):
    text = in_path.read_text(ENC)
    lines = text.splitlines()

    # Prepare per-file output buffer
    out_buf = io.StringIO()

    # Accumulators for one microbatch
    batch_forms: List[List[str]] = []  # list of sentence tokens (forms)
    batch_sent_rows: List[List[str]] = []  # list of token rows (text)
    batch_sent_comments: List[List[str]] = []  # list of comment lines
    batch_fmt: List[str] = []  # per-sentence format type
    pending_count = 0

    def flush_batch():
        nonlocal batch_forms, batch_sent_rows, batch_sent_comments, batch_fmt, pending_count
        if not batch_forms:
            return
        # Forward pass
        doc = nlp(batch_forms)
        # For each sentence, overlay missing fields
        for i, sent in enumerate(doc.sentences):
            fmt = batch_fmt[i]
            needs = needs_layers_for_sentence(batch_sent_rows[i], fmt)
            # Overlay word-by-word
            src_rows = batch_sent_rows[i]
            new_rows: List[str] = []
            words = sent.words
            # Strict 1:1 alignment check
            if len([r for r in src_rows if not r.startswith("#")]) != len(words):
                console.print(
                    f"[warn] Token count mismatch; keeping original sentence in {in_path.name}", style="warn"
                )
                # Output original as-is
                for c in batch_sent_comments[i]:
                    out_buf.write(c + "\n")
                for r in src_rows:
                    out_buf.write(r + "\n")
                out_buf.write("\n")
            else:
                for row, w in zip(src_rows, words):
                    cols = split_cols(row)
                    cols = overlay_token(cols, w, fmt, needs)
                    new_rows.append(join_cols(cols))
                # write sentence
                for c in batch_sent_comments[i]:
                    out_buf.write(c + "\n")
                for r in new_rows:
                    out_buf.write(r + "\n")
                out_buf.write("\n")

        # progress increments
        if total_sents_task is not None:
            progress.advance(total_sents_task, advance=len(batch_forms))
        progress.advance(p_task, advance=0)  # keep bar alive

        # reset accumulators
        batch_forms, batch_sent_rows, batch_sent_comments, batch_fmt = [], [], [], []
        pending_count = 0

    # Iterate sentences
    sent_count = 0
    for comments, toks in iter_sentences_conllu(lines):
        if not toks and comments:
            # document-level comment block; just emit as-is
            for c in comments:
                out_buf.write(c + "\n")
            out_buf.write("\n")
            continue
        if not toks:
            continue

        # determine format
        fmt = "unknown"
        # find first non-comment row
        first_cols = split_cols(toks[0])
        fmt = detect_format(first_cols)

        # Prepare model input (forms)
        forms = []
        for row in toks:
            if row.startswith("#"):
                continue
            cols = split_cols(row)
            if not cols or "-" in cols[0] or "." in cols[0]:
                # Ignore MWEs/empty nodes for alignment input (keep row for output though)
                # For safety, we still push the surface form when numeric ID is integer
                pass
            # FORM is cols[1] for CoNLL-U
            if len(cols) >= 2:
                forms.append(cols[1])
            else:
                forms.append("")

        batch_forms.append(forms)
        batch_sent_rows.append(toks)
        batch_sent_comments.append(comments)
        batch_fmt.append(fmt)
        pending_count += 1
        sent_count += 1

        if pending_count >= sent_batch:
            flush_batch()

    # Final flush
    flush_batch()

    # Atomic write
    out_text = out_buf.getvalue()
    atomic_write_text(out_path, out_text)
    out_buf.close()


# -----------------------
# COMBINE
# -----------------------
def flow_combine(in_dir: Path, out_dir: Path, stats_dir: Path, combined_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    files = list_conllu_files(in_dir)
    target = out_dir / combined_name

    with target.open("w", encoding=ENC) as out:
        for f in files:
            with f.open("r", encoding=ENC) as fh:
                for line in fh:
                    # write line as-is
                    out.write(line)
                if not out.tell():
                    pass
                # ensure sentence break between files
                if not line.endswith("\n"):
                    out.write("\n")
                out.write("\n")
            console.print(f"[ok] Appended {f}", style="ok")

    (stats_dir / "combine_stats.json").write_text(
        json.dumps({"files": len(files), "output": str(target)}, ensure_ascii=False, indent=2), encoding=ENC
    )
    console.print(f"[head]Combined → {target}", style="head")


# -----------------------
# AUGMENT flow
# -----------------------
def flow_augment(in_dir: Path, out_dir: Path, stats_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    files = list_conllu_files(in_dir)
    if not files:
        console.print("[err] No CoNLL-U files found under --in", style="err")
        sys.exit(2)

    # Count sentences for global progress
    total_sents = 0
    per_file_sents: Dict[Path, int] = {}
    console.print("Scanning sentence counts…", style="info")
    for f in files:
        cnt = 0
        for ln in f.read_text(ENC).splitlines():
            if ln.startswith("# sent_id") or ln.strip() == "":
                # we will count by blank delimiters; cheap approximation:
                pass
        # Count by sentence blocks
        blocks = 0
        prev_blank = True
        for ln in f.read_text(ENC).splitlines():
            if not ln.strip():
                if not prev_blank:
                    blocks += 1
                prev_blank = True
            else:
                prev_blank = False
        if not prev_blank:
            blocks += 1
        cnt = blocks
        per_file_sents[f] = cnt
        total_sents += cnt

    prefer_gpu = not args.no_gpu
    batches = parse_batches(args.batches)

    # Build single pipeline to reuse models
    nlp = build_pipeline(args.lang, prefer_gpu, batches)

    # Progress bars
    bar_cols = [
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*bar_cols, console=console) as progress:
        t_all = progress.add_task("[head]All sentences", total=total_sents)
        for f in files:
            t_file = progress.add_task(f"{f.name}", total=per_file_sents[f])

            out_path = out_dir / (f.stem + ".aug.conllu")
            # augment with sub-progress: we pass both tasks to update
            augment_file(
                in_path=f,
                out_path=out_path,
                nlp=nlp,
                sent_batch=args.sent_batch,
                progress=progress,
                p_task=t_file,
                total_sents_task=t_all,
            )
            progress.update(t_file, completed=per_file_sents[f])
            console.print(f"[ok] Augmented {f.name} → {out_path.name}", style="ok")

    # Stats
    (stats_dir / "augment_stats.json").write_text(
        json.dumps({
            "files": len(files),
            "total_sentences": total_sents,
            "lang": args.lang,
            "sent_batch": args.sent_batch,
            "batches": batches,
            "gpu": prefer_gpu
        }, ensure_ascii=False, indent=2),
        encoding=ENC
    )
    console.print("[head]Augment done.", style="head")


# -----------------------
# Main
# -----------------------
def main():
    ap = build_argparser()
    args = ap.parse_args()

    in_dir = Path(args.inp)
    out_dir = Path(args.out)
    stats_dir = Path(args.stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "combine":
        flow_combine(in_dir, out_dir, stats_dir, args.combined_name)
    elif args.mode == "augment":
        flow_augment(in_dir, out_dir, stats_dir, args)
    else:
        console.print("[err] Unknown mode", style="err")
        sys.exit(2)

if __name__ == "__main__":
    main()
