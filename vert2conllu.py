#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VERT → CoNLL-U converter (JANES-style, robust)
- Columns per token line in VERT:
    0 = FORM
    1 = LEMMA
    2 = ORTH/EDIT CODE (ignored)
    3 = XPOS (JOS MSD)
    4 = FEATS (JOS shorthand)
    5.. = tail: "=", CAP, NST, spans like "12-34" → put in MISC
- <g/> denotes a space between tokens. If there is NO <g/> between two tokens,
  add SpaceAfter=No to the PREVIOUS token's MISC.
- <name type="..."> … </name> wraps NE spans → emit NER= B-XXX / I-XXX in MISC.
- All UPOS/HEAD/DEPREL are left "_" (to be filled by Classla augment).
- Writes one .conllu per input .vert (same basename).
"""

import argparse
import re
from pathlib import Path

ENC = "utf-8"

NAME_TYPE_MAP = {
    "per": "PER",
    "loc": "LOC",
    "org": "ORG",
    "misc": "MISC"
}

SPAN_RE = re.compile(r"^\d+-\d+$")

def parse_args():
    ap = argparse.ArgumentParser(description="Convert VERT to CoNLL-U (JANES style).")
    ap.add_argument("--in", dest="inp", required=True,
                    help="Input file or directory containing .vert files")
    ap.add_argument("--out", dest="out", required=True,
                    help="Output directory for .conllu files")
    # Column mapping (0-based)
    ap.add_argument("--col-form", type=int, default=0)
    ap.add_argument("--col-lemma", type=int, default=1)
    ap.add_argument("--col-xpos", type=int, default=3)
    ap.add_argument("--col-featsjos", type=int, default=4)
    ap.add_argument("--col-misc-start", type=int, default=5,
                    help="First index of tail columns that go to MISC (default 5)")
    # Options
    ap.add_argument("--drop-equals", action="store_true",
                    help="Drop bare '=' tails (default: drop them anyway)")
    ap.add_argument("--keep-spans", action="store_true",
                    help="Keep numeric spans (e.g., 12-34) in MISC as Span=12-34")
    ap.add_argument("--glob", default="*.vert",
                    help="Glob to select input files inside a directory (default: *.vert)")
    ap.add_argument("--single", action="store_true",
                    help="If set and input is a dir, write a single combined .conllu (basename: combined.conllu)")
    ap.add_argument("--combined-name", default="combined.conllu",
                    help="Filename for --single output (default: combined.conllu)")
    return ap.parse_args()

def norm_misc_tag(tag: str):
    t = tag.strip()
    if not t or t == "=":
        return None
    if SPAN_RE.match(t):
        return f"Span={t}"
    # keep known flags as-is; otherwise preserve unknown tails
    return t

def start_new_doc(meta_attrs, out, doc_counter):
    # Write doc-level comments
    # Always include deterministic id/comment lines if present
    if meta_attrs:
        if "id" in meta_attrs:
            out.write(f"# newdoc id = {meta_attrs['id']}\n")
        for k, v in meta_attrs.items():
            if k == "id":
                continue
            out.write(f"# {k} = {v}\n")
    else:
        out.write(f"# newdoc id = doc{doc_counter}\n")

def write_sentence(sent_tokens, out, sent_id):
    if not sent_tokens:
        return
    out.write(f"\n# sent_id = {sent_id}\n")
    for tok in sent_tokens:
        # Build 10 CoNLL-U columns
        # ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
        cols = [
            str(tok["id"]),
            tok.get("form", "_"),
            tok.get("lemma", "_"),
            tok.get("upos", "_"),                 # left blank for augment
            tok.get("xpos", "_"),
            tok.get("feats", "_"),
            tok.get("head", "_"),
            tok.get("deprel", "_"),
            tok.get("deps", "_"),
        ]
        misc = tok.get("misc_list", [])
        misc_str = "|".join(misc) if misc else "_"
        cols.append(misc_str)
        out.write("\t".join(cols) + "\n")

def end_sentence_adjust_space_after_no(last_token, saw_gap_after_last):
    # If sentence ends and there was no gap marker after last token,
    # we do nothing; UD does not require SpaceAfter on last token.
    # (We leave it as-is.)
    return

def convert_file(in_path: Path, out_handle, cidx, keep_spans=False):
    """
    Stream-convert a single .vert file to CoNLL-U in out_handle.
    """
    with in_path.open("r", encoding=ENC, errors="replace") as f:
        doc_meta = None
        in_text = False
        in_name = False
        name_type_stack = []  # support nesting just in case
        ner_beg_stack = []    # track beginning per nesting level

        sent_tokens = []
        sent_id_counter = 0
        token_id = 0

        # gap logic: if we DO NOT see <g/> between prev and current token,
        # add SpaceAfter=No to the PREVIOUS token
        gap_after_last = False
        last_token = None

        doc_counter = 0

        for raw in f:
            line = raw.rstrip("\n")

            # TEXT start
            if line.startswith("<text "):
                in_text = True
                doc_counter += 1
                # parse attributes into dict
                attrs = dict(re.findall(r'(\w+)="([^"]*)"', line))
                doc_meta = attrs
                start_new_doc(doc_meta, out_handle, doc_counter)
                continue

            # TEXT end
            if line.strip() == "</text>":
                # flush any open sentence (should not happen if well-formed)
                if sent_tokens:
                    sent_id_counter += 1
                    write_sentence(sent_tokens, out_handle,
                                   f"{doc_meta.get('id','doc'+str(doc_counter))}.s{sent_id_counter}")
                    sent_tokens = []
                in_text = False
                doc_meta = None
                continue

            # NAME (NER) start
            if line.startswith("<name "):
                attrs = dict(re.findall(r'(\w+)="([^"]*)"', line))
                ner_type = attrs.get("type", "").lower()
                ner_label = NAME_TYPE_MAP.get(ner_type, ner_type.upper() if ner_type else "MISC")
                name_type_stack.append(ner_label)
                ner_beg_stack.append(True)
                continue

            # NAME end
            if line.strip() == "</name>":
                if name_type_stack:
                    name_type_stack.pop()
                if ner_beg_stack:
                    ner_beg_stack.pop()
                continue

            # gap marker
            if line.strip() == "<g/>":
                gap_after_last = True
                continue

            # sentence start / end
            if line.strip() == "<s>":
                # reset sentence state
                sent_tokens = []
                sent_id_counter += 1
                token_id = 0
                gap_after_last = False
                last_token = None
                continue

            if line.strip() == "</s>":
                # finalize the sentence
                end_sentence_adjust_space_after_no(last_token, gap_after_last)
                write_sentence(sent_tokens, out_handle,
                               f"{(doc_meta or {}).get('id','doc'+str(doc_counter))}.s{sent_id_counter}")
                sent_tokens = []
                token_id = 0
                gap_after_last = False
                last_token = None
                continue

            # paragraph tags (ignore)
            if line.strip() in ("<p>", "</p>"):
                continue

            # comments or empty
            if not line or line.startswith("#"):
                continue

            # Otherwise: token line (TAB-separated)
            parts = line.split("\t")
            if len(parts) < 2:
                # skip odd lines (e.g., stray markup we didn't handle)
                continue

            # Before adding this token, if previous token exists and there was NO gap,
            # mark SpaceAfter=No on the previous token.
            if last_token is not None and not gap_after_last:
                if "misc_list" not in last_token:
                    last_token["misc_list"] = []
                if "SpaceAfter=No" not in last_token["misc_list"]:
                    last_token["misc_list"].append("SpaceAfter=No")

            # Parse columns
            form = parts[cidx["form"]].strip() if len(parts) > cidx["form"] else "_"
            lemma = parts[cidx["lemma"]].strip() if len(parts) > cidx["lemma"] else "_"
            xpos  = parts[cidx["xpos"]].strip() if len(parts) > cidx["xpos"] else "_"
            feats_raw = parts[cidx["feats"]].strip() if len(parts) > cidx["feats"] else "_"

            # Tail → MISC
            misc_list = []
            tail_start = cidx["misc_start"]
            if tail_start < len(parts):
                for t in parts[tail_start:]:
                    tag = t.strip()
                    if not tag or tag == "=":
                        continue
                    if SPAN_RE.match(tag):
                        if keep_spans:
                            misc_list.append(f"Span={tag}")
                        continue
                    misc_list.append(tag)

            # NER from <name>
            if name_type_stack:
                ner_label = name_type_stack[-1]
                is_begin = ner_beg_stack[-1]
                misc_list.append(f"NER={'B' if is_begin else 'I'}-{ner_label}")
                # set I after first
                ner_beg_stack[-1] = False

            token_id += 1
            tok = {
                "id": token_id,
                "form": form if form else "_",
                "lemma": lemma if lemma else "_",
                "upos": "_",                   # to be filled by classla
                "xpos": xpos if xpos else "_",
                "feats": feats_raw if feats_raw else "_",
                "head": "_",
                "deprel": "_",
                "deps": "_",
                "misc_list": misc_list[:] if misc_list else []
            }
            sent_tokens.append(tok)
            last_token = tok
            # reset gap flag (we only set it when <g/> appears)
            gap_after_last = False

def main():
    args = parse_args()
    in_path = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cidx = {
        "form": args.col_form,
        "lemma": args.col_lemma,
        "xpos": args.col_xpos,
        "feats": args.col_featsjos,
        "misc_start": args.col_misc_start
    }

    if in_path.is_file():
        out_path = (out_dir / in_path.name).with_suffix(".conllu")
        with out_path.open("w", encoding=ENC) as out:
            convert_file(in_path, out, cidx, keep_spans=args.keep_spans)
        print(f"[ok] Converted: {in_path.name} → {out_path.name}")
        return

    # Directory
    files = sorted(in_path.glob(args.glob))
    if not files:
        print(f"[warn] No files matched: {in_path}/{args.glob}")
        return

    if args.single:
        out_path = out_dir / args.combined_name
        with out_path.open("w", encoding=ENC) as out:
            for p in files:
                convert_file(p, out, cidx, keep_spans=args.keep_spans)
        print(f"[ok] Converted (combined): {len(files)} file(s) → {out_path.name}")
        return

    # One output per input
    count = 0
    for p in files:
        out_path = (out_dir / p.name).with_suffix(".conllu")
        with out_path.open("w", encoding=ENC) as out:
            convert_file(p, out, cidx, keep_spans=args.keep_spans)
        count += 1
        print(f"[ok] Converted: {p.name} → {out_path.name}")
    print(f"[done] Converted: {count} file(s)")

if __name__ == "__main__":
    main()
