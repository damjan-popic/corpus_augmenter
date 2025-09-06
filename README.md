Classla All-Processors Pipeline (TEI/CoNLL-U Ready, GPU-Optimized)

This repository provides a robust NLP pipeline for Slovene corpora, built on top of Classla
 (a fork of Stanza specialized for South Slavic languages).

It supports:

Conversion from TEI-encoded corpora (e.g., JANES, GOS) to CoNLL-U skeletons.

Conversion from VRT/Vert formats (optional).

Augmentation of CoNLL-U: filling in missing UPOS, FEATS, HEAD, DEPREL, and NER using Classla.

Combination of multiple CoNLL-U files into a unified corpus.

GPU-aware batch processing with progress reporting.

The design principle: never overwrite existing annotations—only fill in missing ones. This ensures maximal reuse of manually curated annotations while harmonizing corpora.

Installation
# clone the repo
git clone https://github.com/yourusername/classla_tag.git
cd classla_tag

# create a venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install classla rich

# download Slovene models (or replace 'sl' with another language)
python -c "import classla; classla.download('sl')"


If you prefer conda, create a conda environment and install classla + rich inside it.

Scripts
1. tei2conllu_reannotate.py

Convert TEI XML corpora into CoNLL-U skeletons.

Keeps <div type="text" xml:id="..."> as # newdoc id = ....

Extracts <fs><f name="..."> metadata into # text_meta = ....

Converts <w> and <pc> into CoNLL-U rows:

FORM, LEMMA from TEI

XPOS from TEI @ana

UPOS, FEATS, HEAD, DEPREL left blank (_)

Adds NER=B-/I-... spans from <name type="per|org|loc|misc">.

Adds SpaceAfter=No when no <c> (space) follows.

Usage:

python tei2conllu_reannotate.py \
  --in datasets/janes/news_tei \
  --out out_conllu_from_tei

2. run_classla.py

Main driver for Classla-based annotation and augmentation.

Modes:

annotate: process raw text/XML into CoNLL-U

combine: merge multiple CoNLL-U files

augment: fill missing layers in CoNLL-U (7/8/10-column)

Usage:

# augment TEI-converted files
python run_classla.py \
  --mode augment --lang sl \
  --in out_conllu_from_tei \
  --out out_augmented \
  --stats out_stats

# combine multiple conllu into one
python run_classla.py \
  --mode combine \
  --in datasets/sst \
  --out out_combined \
  --stats out_stats \
  --combine combined.conllu

Typical Workflow

Convert TEI → CoNLL-U skeletons

python tei2conllu_reannotate.py \
  --in datasets/janes/news_tei \
  --out out_conllu_from_tei


Augment with Classla

python run_classla.py \
  --mode augment --lang sl \
  --in out_conllu_from_tei \
  --out out_augmented \
  --stats out_stats


(Optional) Merge subcorpora

python run_classla.py \
  --mode combine \
  --in out_augmented \
  --out corpus_full \
  --stats out_stats \
  --combine janes_full.conllu

Output

CoNLL-U files enriched with:

Universal POS (UPOS, English labels: NOUN, VERB, PROPN, …)

Lemmas

Morphological features (FEATS)

Dependency heads and relations (HEAD, DEPREL)

Named entities (NER= tags in MISC)

Stats (out_stats/) with token counts, sentence counts, and layer coverage.

Notes

Classla runs best with a GPU. This repo assumes NVIDIA (tested with RTX 2000 Ada, 16GB).

If running on CPU only, pass --no-gpu.

Conversion scripts are robust to schema variation:

They can process TEI, VRT, and existing CoNLL-U.

They tolerate missing columns (7, 8, or 10 col).

Metadata is preserved in comments for traceability.

The augmenter is idempotent: running it twice won’t overwrite existing filled layers.