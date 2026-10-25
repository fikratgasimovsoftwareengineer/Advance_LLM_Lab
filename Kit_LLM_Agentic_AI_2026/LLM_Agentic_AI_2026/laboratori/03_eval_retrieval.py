"""Laboratorio 3c: HitRate@k su domande con documento di riferimento noto."""
import argparse
import json
from rag_common import ROOT, find_documents, get_resources

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=3)
a = p.parse_args()
if not 1 <= a.k <= 6:
    p.error("--k deve essere compreso tra 1 e 6")

examples = json.loads((ROOT / "dati" / "qa_eval_demo.json").read_text(encoding="utf-8"))
embed, collection = get_resources()
if collection.count() == 0:
    raise SystemExit("Prima indicizzare con python 03_indicizza.py")

hits = 0
for ex in examples:
    found = find_documents(ex["question"], embed, collection, k=a.k)
    retrieved_ids = [d["id"] for d in found]
    hit = ex["expected_doc_id"] in retrieved_ids
    hits += hit
    print(f"{'OK' if hit else 'NO'} expected={ex['expected_doc_id']} got={retrieved_ids} | {ex['question']}")
print(f"HitRate@{a.k}: {hits}/{len(examples)} = {hits/len(examples):.1%}")
print("Nota: dataset illustrativo molto piccolo; nessuna generalizzazione statistica.")
