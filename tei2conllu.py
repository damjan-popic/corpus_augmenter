#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One TEI (with <w>/<pc>/<c>) → one CoNLL-U file.
Minimal columns (FORM, LEMMA, XPOS, MISC SpaceAfter=No). Others are "_".
Auto-detects TEI namespace so <s> are found (prevents empty outputs).
"""

import argparse, re
from pathlib import Path
from typing import List, Tuple, Optional
from xml.etree import ElementTree as ET

ENC = "utf-8"

def detect_ns(root) -> str:
    """Return namespace prefix like '{http://www.tei-c.org/ns/1.0}' or '' if none."""
    m = re.match(r'\{.*\}', root.tag)
    return m.group(0) if m else ''

def text_or_none(el: Optional[ET.Element]) -> Optional[str]:
    return el.text if el is not None else None

def strip_hash(s: Optional[str]) -> Optional[str]:
    if not s: return s
    return s[1:] if s.startswith("#") else s

def read_sentence_tokens(s: ET.Element, TEI: str) -> List[Tuple[str,str,str,bool]]:
    """
    Return tokens for a TEI <s> as (FORM, LEMMA, XPOS, space_after).
    We honor <w>, <pc>, and <c> in document order; nested inside <name> etc. is fine.
    If a following <c> contains whitespace, we *don’t* set SpaceAfter=No.
    """
    items = []
    for el in s.iter():
        tag = el.tag
        if tag == TEI+"w":
            form = (el.text or "").strip()
            if not form:  # skip <w><gap/></w>
                continue
            lemma = el.get("lemma") or "_"
            xpos  = strip_hash(el.get("ana") or "_") or "_"
            items.append(("tok", (form, lemma, xpos)))
        elif tag == TEI+"pc":
            form = (el.text or "").strip()
            if not form: continue
            items.append(("tok", (form, "_", "_")))
        elif tag == TEI+"c":
            items.append(("c", (el.text or "")))

    toks: List[Tuple[str,str,str,bool]] = []
    for i, (kind, payload) in enumerate(items):
        if kind != "tok": continue
        form, lemma, xpos = payload
        space_after = False
        # lookahead to nearest connector
        j = i + 1
        while j < len(items) and items[j][0] not in ("c", "tok"):
            j += 1
        if j < len(items) and items[j][0] == "c":
            ctext = items[j][1]
            if ctext and any(ch.isspace() for ch in ctext):
                space_after = True
        toks.append((form, lemma, xpos, space_after))
    if toks:
        toks[-1] = (toks[-1][0], toks[-1][1], toks[-1][2], False)
    return toks

def sent_to_conllu(tokens: List[Tuple[str,str,str,bool]]) -> List[List[str]]:
    rows = []
    tid = 1
    for form, lemma, xpos, space_after in tokens:
        misc = "_" if space_after else "SpaceAfter=No"
        rows.append([str(tid), form, lemma or "_", "_", xpos or "_", "_", "_", "_", "_", misc])
        tid += 1
    return rows

def write_header(out, xml_path: Path, root, TEI: str):
    out.write(f"# newdoc id = {xml_path.stem}\n")
    # try a few common metadata bits if present
    platform = root.find(".//"+TEI+"f[@name='platform']")
    if platform is not None and platform.text:
        out.write(f"# platform = {platform.text}\n")
    url = root.find(".//"+TEI+"f[@name='url']")
    if url is not None and url.text:
        out.write(f"# url = {url.text}\n")
    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang") or root.get("lang")
    if lang:
        out.write(f"# lang = {lang}\n")

def convert_one(xml_path: Path, out_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    TEI = detect_ns(root)

    sents = root.findall(".//"+TEI+"s")
    with out_path.open("w", encoding=ENC) as out:
        write_header(out, xml_path, root, TEI)
        if not sents:
            out.write("# note = no <s> elements found; check TEI namespace or structure\n\n")
            return 0
        sid = 0
        for s in sents:
            sid += 1
            out.write(f"# sent_id = {xml_path.stem}.s{sid}\n")
            toks = read_sentence_tokens(s, TEI)
            rows = sent_to_conllu(toks)
            for r in rows:
                out.write("\t".join(r) + "\n")
            out.write("\n")
    return sid

def main():
    ap = argparse.ArgumentParser(description="One TEI → one CoNLL-U (no splitting).")
    ap.add_argument("--in", dest="inp", required=True, help="TEI file or folder")
    ap.add_argument("--out", dest="out", required=True, help="Output folder")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    files = [in_path] if in_path.is_file() else sorted(in_path.rglob("*.xml"))
    total_sents = 0; total_files = 0
    for p in files:
        out_path = out_dir / (p.stem + ".conllu")
        try:
            n = convert_one(p, out_path)
            print(f"[ok] {p.name} → {out_path.name}  ({n} sentences)")
            total_sents += n; total_files += 1
        except ET.ParseError as e:
            print(f"[err] {p}: XML parse error → {e}")
        except Exception as e:
            print(f"[err] {p}: {e}")

    print(f"[done] Converted {total_files} file(s), {total_sents} sentence(s) total.")

if __name__ == "__main__":
    main()
