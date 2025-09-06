#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_classla.py — Classla/Stanza Swiss-army tool (annotate/combine/augment)

Features
--------
• annotate : raw text/XML → CoNLL-U (per file) + stats
• combine  : merge .conllu/.conll files → one combined file
• augment  : fill ONLY missing layers in existing CoNLL(-U)
             (lemma, UPOS/XPOS, FEATS, HEAD/DEPREL, NER)
             using pretokenized, no-ssplit pipeline, batched by sentences.

Design
------
• Never overwrites existing non-missing annotations.
• UD 10-col: NER in MISC as NER=B-/I-… (we add when absent).
• JOS/reduced (7–9 col): NER in last col as NER=...
• Robust to 7/8/10-column inputs.
• Sentence-level batching for big speedups on GPU.
"""

from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# ---------- Optional pretty console ----------
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
    from rich.theme import Theme
except Exception:  # rich not installed
    class _D:
        def __call__(self, *a, **k): return self
        def print(self, *a, **k): pass
        def rule(self, *a, **k): pass
        def add_task(self, *a, **k): return 0
        def advance(self, *a, **k): pass
        def update(self, *a, **k): pass
        def remove_task(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    Console = lambda *a, **k: _D()
    Progress = BarColumn = TimeElapsedColumn = TimeRemainingColumn = TextColumn = _D
    Theme = dict

console = Console(theme=Theme({"ok":"green","warn":"yellow","err":"bold red","info":"cyan","dim":"dim"}))

# ---------- NLP ----------
try:
    import classla
except Exception:
    classla = None

# ---------- Constants ----------
ENC = "utf-8"
TEXT_EXTS = {".txt", ".md", ".rtf", ".xml"}  # annotate treats these as raw text
CONL_EXTS = {".conllu", ".conll"}

# ---------- Helpers ----------
def human_bytes(n: int) -> str:
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024 or u == "TB": return f"{n:.1f} {u}"
        n /= 1024

def discover(root: Path, exts: set[str]) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in exts else []
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts])

# ---------- CoNLL structures ----------
class Sent:
    def __init__(self):
        self.comments: List[str] = []
        self.rows: List[List[str]] = []  # token rows

    def token_forms(self) -> List[str]:
        forms = []
        for r in self.rows:
            if r and r[0] and r[0][0].isdigit():
                forms.append(r[1])
        return forms

def parse_conllu(path: Path) -> List[Sent]:
    sents: List[Sent] = []
    cur = Sent()
    with path.open("r", encoding=ENC, errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if cur.rows or cur.comments:
                    sents.append(cur); cur = Sent()
                continue
            if line.startswith("#"):
                cur.comments.append(line.rstrip("\n"))
            else:
                cur.rows.append(line.rstrip("\n").split("\t"))
    if cur.rows or cur.comments:
        sents.append(cur)
    return sents

def write_conllu_sents(sents: List[Sent], out_path: Path):
    with out_path.open("w", encoding=ENC) as out:
        for s in sents:
            for c in s.comments:
                out.write(c + "\n")
            for r in s.rows:
                out.write("\t".join(r) + "\n")
            out.write("\n")

# ---------- Stats ----------
def quick_stats(sents: List[Sent]) -> dict:
    sN = len(sents); tok = 0; upos=Counter(); xpos=Counter(); deprel=Counter()
    for s in sents:
        for r in s.rows:
            if r and r[0] and r[0][0].isdigit():
                tok += 1
                if len(r)>3: upos[r[3]] += 1
                if len(r)>4: xpos[r[4]] += 1
                if len(r)>7: deprel[r[7]] += 1
    avg = round(tok/sN, 3) if sN else 0.0
    return {"sentences": sN, "tokens": tok, "avg_sent_len_words": avg, "upos": upos, "xpos": xpos, "deprel": deprel}

def write_corpus_summaries(totals: dict, st_dir: Path, label: str):
    st_dir.mkdir(parents=True, exist_ok=True)
    csv_path = st_dir / f"{label}_corpus_stats.csv"
    with csv_path.open("w", newline="", encoding=ENC) as f:
        w = csv.writer(f)
        w.writerow(["files","bytes","sentences","tokens"])
        w.writerow([totals.get("files",0), totals.get("bytes",0), totals.get("sentences",0), totals.get("tokens",0)])
    md_path = st_dir / f"{label}_corpus_summary.md"
    with md_path.open("w", encoding=ENC) as f:
        f.write(f"# Corpus Summary ({label})\n\n")
        f.write(f"**Files**: {totals.get('files',0)}  ")
        f.write(f"**Bytes**: {human_bytes(totals.get('bytes',0))}  ")
        f.write(f"**Sentences**: {totals.get('sentences',0)}  ")
        f.write(f"**Tokens**: {totals.get('tokens',0)}\n")
    console.print(f"[ok]Wrote summaries → {csv_path}, {md_path}")

# ---------- Format + need detection ----------
def schema_of(sent: Sent) -> str:
    for r in sent.rows:
        return 'ud' if len(r) >= 10 else 'reduced'
    return 'ud'

def needs_layers(sent: Sent) -> dict:
    need = {k: False for k in ("lemma","pos","feats","deprel","ner")}
    sch = schema_of(sent)
    for r in sent.rows:
        if not r or not r[0] or not r[0][0].isdigit():
            continue
        if len(r) < 3 or r[2] in ('_', ''): need['lemma'] = True
        if len(r) < 5 or r[3] in ('_', '') or r[4] in ('_', ''): need['pos'] = True
        if len(r) < 6 or r[5] in ('_', ''): need['feats'] = True
        if len(r) < 8 or r[6] in ('_', '') or r[7] in ('_', ''): need['deprel'] = True
        if sch=='ud':
            ok = (len(r)>=10 and r[9] not in ('','_') and 'NER=' in r[9] and 'NER=O' not in r[9])
            if not ok: need['ner'] = True
        else:
            last = r[-1] if r else ''
            ok = (last not in ('','_') and 'NER=' in last and 'NER=O' not in last)
            if not ok: need['ner'] = True
    return need

# ---------- Overlay helpers ----------
def overlay_ud(row: List[str], w, t, need: dict):
    if need.get('lemma'):
        if len(row) < 3: row.extend(['_']*(3-len(row)))
        if row[2] in ('_', '') and getattr(w, 'lemma', None):
            row[2] = w.lemma
    if need.get('pos'):
        if len(row) < 5: row.extend(['_']*(5-len(row)))
        if row[3] in ('_', '') and getattr(w, 'upos', None):
            row[3] = w.upos
        if row[4] in ('_', '') and getattr(w, 'xpos', None):
            row[4] = w.xpos
    if need.get('feats'):
        if len(row) < 6: row.extend(['_']*(6-len(row)))
        if row[5] in ('_', '') and getattr(w, 'feats', None):
            row[5] = w.feats
    if need.get('deprel'):
        if len(row) < 8: row.extend(['_']*(8-len(row)))
        if row[6] in ('_', '') and getattr(w, 'head', None):
            row[6] = str(w.head)
        if row[7] in ('_', '') and getattr(w, 'deprel', None):
            row[7] = w.deprel
    if need.get('ner'):
        if len(row) < 10: row.extend(['_']*(10-len(row)))
        misc = row[9] if row[9] not in ('','_') else ''
        ner_tag = getattr(t, 'ner', None)
        if ner_tag and (('NER=' not in misc) or ('NER=O' in misc)):
            misc = re.sub(r'(?:^|\|)NER=O(?:$|\|)', '', misc).strip('|')
            row[9] = (misc + '|' if misc else '') + f'NER={ner_tag}'

def overlay_reduced(row: List[str], w, t, need: dict):
    cols = len(row)
    if need.get('lemma') and cols >= 3 and row[2] in ('_', '') and getattr(w, 'lemma', None): row[2] = w.lemma
    if need.get('pos'):
        if cols >= 4 and row[3] in ('_', '') and getattr(w, 'upos', None): row[3] = w.upos
        if cols >= 5 and row[4] in ('_', '') and getattr(w, 'xpos', None): row[4] = w.xpos
    if need.get('feats') and cols >= 6 and row[5] in ('_', '') and getattr(w, 'feats', None): row[5] = w.feats
    if need.get('deprel'):
        if cols >= 7 and row[6] in ('_', '') and getattr(w, 'head', None): row[6] = str(w.head)
        if cols >= 8 and row[7] in ('_', '') and getattr(w, 'deprel', None): row[7] = w.deprel
    if need.get('ner') and cols >= 1:
        ner_tag = getattr(t, 'ner', None)
        if ner_tag and (row[-1] in ('','_') or ('NER=' not in row[-1]) or ('NER=O' in row[-1])):
            row[-1] = f'NER={ner_tag}'

# ---------- Augment (batched) ----------
def augment_batch(sent_batch: List[Sent], nlp) -> List[Tuple[int,int,int,int,int]]:
    """Run Classla once on a list of pretokenized sentences. Return per-sentence change counts."""
    results = [(0,0,0,0,0)] * len(sent_batch)
    inputs, idx_map = [], []
    for i, s in enumerate(sent_batch):
        forms = s.token_forms()
        if forms:
            inputs.append(forms)
            idx_map.append(i)
    if not inputs:
        return results

    doc = nlp(inputs)  # one forward pass
    for di, cl_sent in enumerate(doc.sentences):
        si = idx_map[di]
        s = sent_batch[si]
        tokens = cl_sent.tokens
        words  = cl_sent.words

        forms = s.token_forms()
        pred  = [t.text for t in tokens]
        if forms != pred:
            console.print(f"[warn]Token mismatch in pretokenized mode; skipping.\n  SRC: {forms}\n  NLP: {pred}")
            continue

        sch = schema_of(s)
        need = needs_layers(s)
        if not any(need.values()):
            continue

        changed = [0,0,0,0,0]
        src_rows = [r for r in s.rows if r and r[0] and r[0][0].isdigit()]
        for i_tok, row in enumerate(src_rows):
            w = words[i_tok]
            t = tokens[i_tok]
            before = row.copy()
            if sch == 'ud':
                overlay_ud(row, w, t, need)
            else:
                overlay_reduced(row, w, t, need)
            if need.get('lemma') and len(before)>2 and before[2]!=row[2]: changed[0]+=1
            if need.get('pos') and ((len(before)>3 and before[3]!=row[3]) or (len(before)>4 and before[4]!=row[4])): changed[1]+=1
            if need.get('feats') and len(before)>5 and before[5]!=row[5]: changed[2]+=1
            if need.get('deprel') and ((len(before)>6 and before[6]!=row[6]) or (len(before)>7 and before[7]!=row[7])): changed[3]+=1
            if need.get('ner'):
                if sch=='ud':
                    if len(before)>9 and before[9]!=row[9]: changed[4]+=1
                else:
                    if before[-1]!=row[-1]: changed[4]+=1
        results[si] = tuple(changed)
    return results

# ---------- Pipelines ----------
def build_annotate_pipeline(lang: str, use_gpu: bool, batches: dict):
    assert classla is not None, 'Install classla: pip install classla'
    try: classla.download(lang)
    except Exception: pass
    return classla.Pipeline(
        lang=lang,
        processors='tokenize,mwt,pos,lemma,depparse,ner',
        use_gpu=use_gpu,
        tokenize_batch_size=batches.get('tokenize',10000),
        pos_batch_size=batches.get('pos',1000),
        lemma_batch_size=batches.get('lemma',5000),
        depparse_batch_size=batches.get('depparse',5000),
        ner_batch_size=batches.get('ner',200),
    )

def build_augment_pipeline(lang: str, use_gpu: bool, batches: dict):
    assert classla is not None, 'Install classla: pip install classla'
    try: classla.download(lang)
    except Exception: pass
    return classla.Pipeline(
        lang=lang,
        processors='tokenize,pos,lemma,depparse,ner',  # pretokenized; no MWT
        use_gpu=use_gpu,
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
        tokenize_batch_size=batches.get('tokenize',20000),
        pos_batch_size=batches.get('pos',4000),
        lemma_batch_size=batches.get('lemma',10000),
        depparse_batch_size=batches.get('depparse',6000),
        ner_batch_size=batches.get('ner',1000),
    )

# ---------- Fast sentence counter for global progress ----------
def count_conllu_sentences(path: Path) -> int:
    """Fast sentence counter without full parse (counts blank-line boundaries)."""
    sents = 0
    in_sent = False
    with path.open("r", encoding=ENC, errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if in_sent:
                    sents += 1
                    in_sent = False
            elif not line.startswith("#"):
                in_sent = True
    if in_sent:
        sents += 1
    return sents

# ---------- Flows ----------
def flow_annotate(files: List[Path], out_dir: Path, st_dir: Path, args):
    use_gpu = not args.no_gpu
    console.print('[info]Initializing annotate pipeline…')
    nlp = build_annotate_pipeline(args.lang, use_gpu, args.batch_sizes)
    console.print(f'[ok]Pipeline ready → {nlp.processors}')
    totals = {"files":0,"bytes":0,"sentences":0,"tokens":0}
    with Progress(TextColumn('[progress.description]{task.description}'), BarColumn(),
                  TextColumn('{task.percentage:>3.0f}%'), TimeElapsedColumn(),
                  TimeRemainingColumn(), console=console) as prog:
        task = prog.add_task('Annotating', total=len(files))
        for path in files:
            raw = path.read_text(ENC, errors='ignore')
            doc = nlp(raw)
            out_path = out_dir / (path.stem + '.conllu')
            out_path.write_text(doc.to_conll(), encoding=ENC)
            ds = quick_stats(parse_conllu(out_path))
            totals["files"]+=1; totals["bytes"]+=path.stat().st_size
            totals["sentences"]+=ds["sentences"]; totals["tokens"]+=ds["tokens"]
            console.print(f"[ok]{path.name} → {out_path.name}")
            prog.advance(task)
    write_corpus_summaries(totals, st_dir, label="annotate")

def flow_combine(files: List[Path], combined_out: Path):
    if combined_out.exists(): combined_out.unlink()
    with combined_out.open('w', encoding=ENC) as out, \
         Progress(TextColumn('[progress.description]{task.description}'), BarColumn(),
                  TextColumn('{task.percentage:>3.0f}%'), TimeElapsedColumn(),
                  TimeRemainingColumn(), console=console) as prog:
        task = prog.add_task('Combining', total=len(files))
        for p in files:
            sents = parse_conllu(p)
            if sents and not any(c.lower().startswith('# newdoc') for c in sents[0].comments):
                out.write(f'# newdoc id = {p.stem}\n')
            for s in sents:
                for c in s.comments:
                    out.write(c + '\n')
                for r in s.rows:
                    out.write('\t'.join(r) + '\n')  # <<< FIXED NEWLINE
                out.write('\n')
            console.print(f"[ok]{p.name} → appended to {combined_out.name}")
            prog.advance(task)

def flow_augment(files: List[Path], out_dir: Path, st_dir: Path, args):
    use_gpu = not args.no_gpu
    console.print('[info]Initializing augment pipeline (pretokenized + batched)…')
    nlp = build_augment_pipeline(args.lang, use_gpu, args.batch_sizes)
    console.print(f'[ok]Augment pipeline ready → {nlp.processors}')

    # Global sentence count for incremental progress
    console.print('[info]Scanning sentence counts…')
    total_sents_all = sum(count_conllu_sentences(p) for p in files)
    console.print(f"[info]Total sentences to process: {total_sents_all:,}")

    totals = {"files":0,"bytes":0,"sentences":0,"tokens":0}
    B = max(1, int(args.sent_batch))

    with Progress(
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TextColumn('{task.completed:>9}/{task.total:<9}'),
        TextColumn('{task.percentage:>4.0f}%'),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as prog:

        global_task = prog.add_task('All sentences', total=total_sents_all)
        files_task  = prog.add_task('Files', total=len(files))

        for path in files:
            sents = parse_conllu(path)
            file_task = prog.add_task(f"[dim]{path.name}[/dim]", total=len(sents))

            updates = {"lemma":0,"pos":0,"feats":0,"deprel":0,"ner":0}
            for i in range(0, len(sents), B):
                batch = sents[i:i+B]
                deltas = augment_batch(batch, nlp)
                processed = len(deltas)
                for d in deltas:
                    updates["lemma"]+=d[0]; updates["pos"]+=d[1]; updates["feats"]+=d[2]; updates["deprel"]+=d[3]; updates["ner"]+=d[4]
                prog.advance(file_task, processed)
                prog.advance(global_task, processed)

            out_path = out_dir / path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_conllu_sents(sents, out_path)

            ds = quick_stats(sents)
            totals["files"]+=1; totals["bytes"]+=path.stat().st_size
            totals["sentences"]+=ds["sentences"]; totals["tokens"]+=ds["tokens"]
            with (st_dir / (path.stem + '.augment.json')).open('w', encoding=ENC) as f:
                json.dump({"path":str(path), "updated":updates, **ds}, f, ensure_ascii=False, indent=2)

            console.print(f"[ok]{path.name} → augmented → {out_path.name}  "
                          f"+NER:{updates['ner']}  +POS:{updates['pos']}  "
                          f"+LEM:{updates['lemma']}  +DEPS:{updates['deprel']}")

            prog.advance(files_task, 1)
            prog.remove_task(file_task)

    write_corpus_summaries(totals, st_dir, label="augment")

# ---------- CLI ----------
def parse_batches(spec: str) -> dict:
    out = {}
    for pair in spec.split(','):
        if not pair: continue
        k,v = pair.split('='); out[k.strip()] = int(v)
    return out

def main():
    ap = argparse.ArgumentParser(description='Classla + CoNLL-U (annotate/combine/augment)')
    ap.add_argument('--mode', choices=['annotate','combine','augment'], required=True)
    ap.add_argument('--in', dest='inp', required=True, help='Input folder or file')
    ap.add_argument('--out', dest='out', required=True, help='Output folder')
    ap.add_argument('--stats', dest='stats', required=True, help='Stats folder')
    ap.add_argument('--lang', default='sl', help='ISO code (e.g., sl)')
    ap.add_argument('--no-gpu', action='store_true', help='Force CPU')
    ap.add_argument('--batches', default='tokenize=20000,pos=4000,lemma=10000,depparse=6000,ner=1000',
                    help='Comma-separated batch sizes per processor')
    ap.add_argument('--combine', default=None, help='In augment: combine all inputs first into this filename')
    ap.add_argument('--combined-name', default='combined.conllu', help='Name for combine mode output')
    ap.add_argument('--sent-batch', type=int, default=128, help='Sentences per augmentation batch')
    args = ap.parse_args()

    in_path = Path(args.inp); out_dir = Path(args.out); st_dir = Path(args.stats)
    out_dir.mkdir(parents=True, exist_ok=True); st_dir.mkdir(parents=True, exist_ok=True)
    args.batch_sizes = parse_batches(args.batches)

    console.rule('[bold cyan]Classla / CoNLL-U Tool')

    if args.mode == 'annotate':
        files = discover(in_path, TEXT_EXTS)
        if not files: console.print('[err]No raw text/XML found.'); sys.exit(1)
        flow_annotate(files, out_dir, st_dir, args)

    elif args.mode == 'combine':
        files = discover(in_path, CONL_EXTS)
        if not files: console.print('[err]No CoNLL-U files to combine.'); sys.exit(1)
        flow_combine(files, out_dir / args.combined_name)

    elif args.mode == 'augment':
        if args.combine:
            files = discover(in_path, CONL_EXTS)
            if not files: console.print('[err]No CoNLL-U files to combine/augment.'); sys.exit(1)
            combined = out_dir / args.combine
            flow_combine(files, combined)
            targets = [combined]
        else:
            targets = discover(in_path, CONL_EXTS)
            if not targets and in_path.is_file() and in_path.suffix.lower() in CONL_EXTS:
                targets = [in_path]
            if not targets: console.print('[err]No CoNLL-U files to augment.'); sys.exit(1)
        flow_augment(targets, out_dir, st_dir, args)

    console.rule('[bold green]Done')

if __name__ == '__main__':
    main()
