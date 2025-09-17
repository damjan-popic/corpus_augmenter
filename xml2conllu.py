#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xml2conllu.py — Convert simple SketchEngine-like XML to basic CoNLL-U.

Input structure:
<corpus>
  <doc title="..." author="..." ...>
    line of text...
    (blank lines allowed)
  </doc>
  ...
</corpus>

Rules:
- Each NON-EMPTY line becomes one sentence.
- Tokenization: Unicode words and single punctuation chars.
- SpaceAfter=No preserved by inspecting original spacing.
- All columns except FORM are '_' (to be filled by your augment pipeline).

Usage:
  python xml2conllu.py --in corpus.xml --out out_converted/conversion.conllu
  # or write next to the input (auto .conllu name)
"""

from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET

WORD_PUNCT = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def iter_tokens_with_space(text: str):
    """Yield (token, space_after_no: bool) from a line, preserving 'SpaceAfter=No'."""
    for m in WORD_PUNCT.finditer(text):
        tok = m.group(0)
        end = m.end()
        # SpaceAfter=No if there is a next char and it's NOT whitespace
        space_after_no = (end < len(text)) and (not text[end].isspace())
        yield tok, space_after_no

def sanitize_doc_id(value: str) -> str:
    base = re.sub(r"\s+", "_", value.strip())
    base = re.sub(r"[^0-9A-Za-z_\-\.]+", "", base)
    return base[:80] or "doc"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input XML file")
    ap.add_argument("--out", dest="out", default=None, help="Output .conllu file (default: alongside input)")
    ap.add_argument("--encoding", default="utf-8", help="XML file encoding (default: utf-8)")
    ap.add_argument("--sent-per-line", action="store_true", default=True,
                    help="Treat each non-empty line as a sentence (default: on)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".conllu")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse XML
    try:
        tree = ET.parse(in_path)
    except ET.ParseError as e:
        print(f"XML parse error: {e}", file=sys.stderr)
        sys.exit(1)

    root = tree.getroot()
    if root.tag.lower() != "corpus":
        print("Root element is not <corpus>.", file=sys.stderr)

    sent_global = 0
    doc_idx = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for doc in root.findall(".//doc"):
            doc_idx += 1
            attrs = {k: (v or "").strip() for k, v in doc.attrib.items()}
            # newdoc id
            title = attrs.get("title") or f"doc{doc_idx}"
            newdoc_id = f"{doc_idx:06d}.{sanitize_doc_id(title)}"
            out.write(f"# newdoc id = {newdoc_id}\n")
            # dump all attributes as comments
            for k, v in attrs.items():
                out.write(f"# {k} = {v}\n")
            out.write("\n")

            # Text content: preserve lines
            text = (doc.text or "")
            # Also include tails of nested tags, if any (rare in this format)
            for elem in doc:
                if elem.text:
                    text += elem.text
                if elem.tail:
                    text += elem.tail

            # Normalize newlines, iterate lines
            for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                line = raw_line.strip()
                if not line:
                    continue  # keep blank lines as doc separators but no empty sentences
                sent_global += 1
                sent_id = f"{newdoc_id}.s{sent_global}"
                out.write(f"# sent_id = {sent_id}\n")
                out.write(f"# text = {raw_line}\n")

                tid = 1
                for tok, space_no in iter_tokens_with_space(raw_line):
                    cols = [
                        str(tid),        # ID
                        tok,             # FORM
                        "_",             # LEMMA
                        "_",             # UPOS
                        "_",             # XPOS
                        "_",             # FEATS
                        "_",             # HEAD
                        "_",             # DEPREL
                        "_",             # DEPS
                        "_"              # MISC (will add SpaceAfter=No if needed)
                    ]
                    if space_no:
                        cols[9] = "SpaceAfter=No"
                    out.write("\t".join(cols) + "\n")
                    tid += 1
                out.write("\n")

    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
