#!/usr/bin/env python3
from pathlib import Path

# adjust paths
vert_path = Path("datasets/janes/vert/janes_tweet.vert.enc")
conllu_path = Path("out_conllu_from_vert/janes.tweet.conllu")

vert_tokens = 0
vert_sents = 0
with vert_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("TOKEN"):
            vert_tokens += 1
        elif line.startswith("<s>"):
            vert_sents += 1

conllu_tokens = 0
conllu_sents = 0
with conllu_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            conllu_tokens += 1
        elif line.startswith("# sent_id"):
            conllu_sents += 1

print("VERT   → tokens:", vert_tokens, "sentences:", vert_sents)
print("CONLLU → tokens:", conllu_tokens, "sentences:", conllu_sents)
