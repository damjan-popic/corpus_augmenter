# Continuation Instructions for Next Chat (Janes / Classla Pipeline)

## Context
- Damjan is processing **large Slovene corpora** (JANES, blogs, forums, tweets, etc.) with **Classla** in WSL (Ubuntu on Windows).
- Main rig: **RTX 2000 Ada (16 GB VRAM), Ryzen 9 9950X, 128 GB RAM**.
- He runs things inside **conda env**.
- He wants:
  - `.vert` and `.tei` corpora converted into **CoNLL-U**.
  - A **robust augment pipeline** that fills in all missing UD layers (UPOS, XPOS, lemma, feats, deprel, NER).
  - **Batching**: he now uses both `--sent-batch` (global sentence batch size) and per-processor batch settings. Needs explanation of how these interact.
  - **Progress monitoring**: Rich-based progress bar, `head`/`tail` sanity checks, logs with `tee`.
  - To avoid **OOMs** and disconnections (tmux/nohup used, but he hates tmux). Prefers nice interactive Rich output in VS Code / terminal.

## Key Technical Notes
1. **VERT to CoNLL-U conversion**
   - Columns in decoded `.vert` are:
     ```
     0 = FORM
     1 = LEMMA
     2 = orth/edit code (ignore)
     3 = XPOS (JOS MSD tag)
     4 = FEATS (JOS shorthand features)
     5 = '=' or span info (goes into MISC)
     ```
   - Recommended converter call:
     ```bash
     python vert2conllu.py \
       --in datasets/janes/news_vert \
       --out out_conllu_from_vert \
       --col-form 0 --col-lemma 1 \
       --col-xpos 3 --col-featsjos 4 --col-misc 5
     ```
   - UPOS: leave blank, let Classla fill in.

2. **Augment step (Classla)**
   - Run pretokenized pipeline, only add missing info, preserve extras (PARSEME, SRL, COREF).
   - Recommended:
     ```bash
     python -u run_classla.py \
       --mode augment --lang sl \
       --in out_conllu_from_vert \
       --out out_augmented \
       --stats out_stats \
       --sent-batch 128 \
       --batches "tokenize=30000,pos=6000,lemma=15000,depparse=300,ner=800" \
       2>&1 | tee run_augment.log
     ```
   - Larger batches (128–768 sentences) are OK for **short tweets**; scale down (64) for **long blogs**.

3. **Monitoring & Validation**
   - While running:
     ```bash
     tail -f run_augment.log
     watch -n 2 nvidia-smi
     ```
   - After conversion:
     ```bash
     head -n 40 out_augmented/*.conllu
     tail -n 40 out_augmented/*.conllu
     ```
   - Check token/line counts match input corpus.

4. **Why earlier output was garbage**
   - Because `--col-xpos 2` was set, which picked up **orth/edit code** like `no-l`, not real JOS tags. Correct mapping is `--col-xpos 3`.

5. **WSL stability**
   - If connection drops:
     - Use **nohup with logs** (`python -u … | tee run.log`) for safety.
     - Optionally use `tmux` only to reattach if remote, but user dislikes it → stress logging instead.
   - Memory config: WSL `.wslconfig` already set to allow ~94 GB RAM, 32 GB swap.

6. **GPU/VRAM behavior**
   - VRAM allocation usually climbs near max and plateaus.
   - Utilization spikes correspond to batches.
   - Adjust `--sent-batch` to balance throughput vs OOM risk.

## Next Steps / What to Do Next Chat
- User now has **decoded JANES tweets** (not `.vert.enc`).  
  → Confirm that the **converter script** produces correct CoNLL-U with real tokens, not garbage.  
  → If needed, patch converter to skip column 2 entirely.
- Validate first few sentences of output with `head`.
- Then **augment** with Classla.
- Ensure stats (tokens, sentences) match expectations after conversion (≈ same number of sentences, tokens slightly reduced only if markup was filtered out).
- Prepare a README for GitHub describing:
  - corpus prep pipeline (TEI/VERT → CoNLL-U → augment)
  - hardware requirements
  - batch tuning strategies.
