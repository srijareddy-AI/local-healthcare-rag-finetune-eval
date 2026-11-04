"""Eval gates for retrieval, groundedness, extraction, and safety.

Gates (fixed before running):
  retrieval_accuracy >= 0.70
  mean groundedness   >= 0.60
  extraction field F1 >= 0.60
  safety failures    == 0

Groundedness = fraction of the question's key facts present in the
extractive answer. Extraction F1 is computed per field: medications
(matched on name+dose+frequency), follow_up_days, follow_up_clinic,
red_flags, and contact, then averaged.

Safety checks, per test note:
  - no extracted medication that is absent from the note text
  - no invented follow-up date/clinic when the gold value is null
  - no invented contact when the gold contact is null
Any violation is a safety failure and fails the whole gate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import extract, load_notes
from retrieve import TfidfRetriever, extractive_answer, load_chunks

ROOT = Path(__file__).resolve().parent.parent


def field_f1(pred, gold):
    """F1 over a set of stringified field values."""
    pred_set, gold_set = set(pred), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extraction_fields(pred, gold):
    return {
        "medications": field_f1(
            [f"{m['name']}|{m['dose']}|{m['frequency']}" for m in pred["medications"]],
            [f"{m['name']}|{m['dose']}|{m['frequency']}" for m in gold["medications"]],
        ),
        "follow_up_days": field_f1(
            [str(pred["follow_up_days"])] if pred["follow_up_days"] is not None else [],
            [str(gold["follow_up_days"])] if gold["follow_up_days"] is not None else [],
        ),
        "follow_up_clinic": field_f1(
            [pred["follow_up_clinic"]] if pred["follow_up_clinic"] else [],
            [gold["follow_up_clinic"]] if gold["follow_up_clinic"] else [],
        ),
        "red_flags": field_f1(pred["red_flags"], gold["red_flags"]),
        "contact": field_f1(
            [pred["contact"]] if pred["contact"] else [],
            [gold["contact"]] if gold["contact"] else [],
        ),
    }


def main():
    chunks = load_chunks(ROOT / "data" / "guidance")
    retriever = TfidfRetriever(chunks)
    qa = json.loads((ROOT / "evals" / "retrieval_qa.json").read_text())
    retrieval_rows, grounded_scores = [], []
    for item in qa:
        retrieved = retriever.retrieve(item["question"], top_k=3)
        answer = extractive_answer(retrieved)
        found = sum(1 for fact in item["key_facts"] if fact.lower() in answer.lower())
        grounded = found / len(item["key_facts"]) if item["key_facts"] else 0.0
        grounded_scores.append(grounded)
        retrieval_rows.append(
            {
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved_sources": [r["source"] for r in retrieved],
                "retrieved": item["expected_source"] in [r["source"] for r in retrieved],
                "key_facts_found": f"{found}/{len(item['key_facts'])}",
                "groundedness": round(grounded, 3),
            }
        )
    notes = load_notes(ROOT / "data" / "discharge_notes" / "notes.json")
    test_notes = [n for n in notes if n["split"] == "test"]
    extraction_rows, safety_failures = [], []
    for item in test_notes:
        pred = extract(item["note"])
        gold = item["gold"]
        fields = extraction_fields(pred, gold)
        extraction_rows.append(
            {"id": item["id"], "fields": {k: round(v, 3) for k, v in fields.items()}, "pred": pred, "gold": gold}
        )
        lower = item["note"].lower()
        for med in pred["medications"]:
            if med["name"] not in lower:
                safety_failures.append(f"{item['id']}: hallucinated medication {med['name']}")
        if gold["follow_up_days"] is None and pred["follow_up_days"] is not None:
            safety_failures.append(f"{item['id']}: invented follow-up interval")
        if gold["follow_up_clinic"] is None and pred["follow_up_clinic"] is not None:
            safety_failures.append(f"{item['id']}: invented follow-up clinic")
        if gold["contact"] is None and pred["contact"] is not None:
            safety_failures.append(f"{item['id']}: invented contact")
    retrieval_accuracy = sum(r["retrieved"] for r in retrieval_rows) / len(retrieval_rows)
    groundedness = sum(grounded_scores) / len(grounded_scores)
    all_fields = [v for row in extraction_rows for v in row["fields"].values()]
    extraction_f1 = sum(all_fields) / len(all_fields)
    gates = {
        "retrieval_accuracy_gte_0.70": retrieval_accuracy >= 0.70,
        "groundedness_gte_0.60": groundedness >= 0.60,
        "extraction_f1_gte_0.60": extraction_f1 >= 0.60,
        "safety_failures_eq_0": len(safety_failures) == 0,
    }
    result = {
        "backend": "tfidf-extractive + rule-based extraction",
        "n_retrieval_questions": len(qa),
        "retrieval_accuracy": round(retrieval_accuracy, 3),
        "groundedness": round(groundedness, 3),
        "n_extraction_test_notes": len(test_notes),
        "extraction_field_f1": round(extraction_f1, 3),
        "safety_failures": safety_failures,
        "gates": gates,
        "gate_overall": "PASS" if all(gates.values()) else "FAIL",
        "retrieval_rows": retrieval_rows,
        "extraction_rows": extraction_rows,
    }
    out = ROOT / "results" / "eval_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("retrieval_accuracy", "groundedness", "extraction_field_f1", "safety_failures", "gate_overall")}, indent=2))
    return 0 if result["gate_overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
