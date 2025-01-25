import json
import re
import sys, os

from normalizers.english import EnglishTextNormalizer


normaliser = EnglishTextNormalizer()
normalise = True

def norm(val):
    if normalise:
        return normaliser(val)
    else:
        return val

with open(infile) as fin:
    output = json.load(fin)

id2files = {}
with open("evaluation/ref_2000_zeroshot.jsonl") as fin:
    for line in fin:
        data = json.loads(line)
        id2files[data["slurp_id"]] = [rec["file"] for rec in data["recordings"]]
