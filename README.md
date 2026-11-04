# Local Healthcare Care-Transition RAG with Fine-Tuned Extraction and Safety Eval Gates

> **For research and demo only. Synthetic, de-identified data only. Not medical
> advice. This project does not diagnose, recommend treatment, or handle any
> real patient. Human clinician review is required before any real use.**

## The problem

Care transitions fail when discharge information is unstructured and
inconsistent. A medication list, a follow-up interval, or a red-flag symptom
can be present in a discharge note but lost in the handoff — and that loss
contributes to avoidable readmissions. This project tackles the *information
handoff* problem, not diagnosis: can a local system reliably extract the
structured handoff (medications, follow-up timing, red-flag symptoms, care-team
contact) from a discharge note, ground its answers in retrieved guidance, and
refuse to invent what is missing?

## Results up front (real, measured, reproducible)

Measured baseline — TF-IDF/extractive retrieval + rule-based extraction,
10 retrieval questions, 4 held-out synthetic test notes:

| Metric | Result | Gate |
| --- | --- | --- |
| Retrieval accuracy | **1.000** (10/10 questions retrieved the expected guidance doc) | ≥ 0.70 |
| Groundedness (key-fact overlap) | **0.867** | ≥ 0.60 |
| Structured-extraction field F1 | **1.000** (all 5 fields, all 4 test notes) | ≥ 0.60 |
| Safety failures | **0** | == 0 |
| **Gate overall** | **PASS** | |

Full rows: [`results/eval_results.json`](results/eval_results.json). Run it
yourself: `python3 src/evaluate.py` (standard library only — no install
needed for the baseline).

Optional local-LLM path — Ollama `llama3.2:1b`, temperature 0, 2 notes:

| Note | Mean field F1 | Note |
| --- | --- | --- |
| SYN-009 (heart failure) | 0.80 | Missed `follow_up_days` (returned null, gold 7) |
| SYN-011 (nothing documented) | 1.00 | After a safety-filter fix (see below) |
| **Mean** | **0.90** | Full rows: [`results/ollama_extraction.json`](results/ollama_extraction.json) |

Honest caveat on the 1.000 extraction score: the synthetic notes use a
consistent section format, so the rule-based extractor is expected to do well
on them. That score proves the pipeline, schema, and safety gates work — it
does **not** claim real-world clinical-note performance, which would need
messier, real-format notes and clinician-validated gold labels.

### A safety flaw the gate caught

The first Ollama run on SYN-011 ("Medications: none documented") returned
placeholder medications named "none". The safety filter in `src/generate.py`
now drops placeholder entries (`none`, `not documented`, null doses) and
normalises missing red flags to `[]`. After the fix, SYN-011 extracts
perfectly. This is reported, not hidden, because a safety gate that never
catches anything has not been tested.

## Architecture

```
Synthetic discharge note
        │
        ▼
┌──────────────────────┐      ┌─────────────────────────────┐
│ Rule-based extractor │      │ TF-IDF retrieval (stdlib)   │
│ src/extract.py       │      │ src/retrieve.py over public │
│  → fixed schema      │      │ guidance summaries          │
│  → null, never guess │      │  → extractive answer only   │
└──────────┬───────────┘      └──────────────┬──────────────┘
           │                                 │
           ▼                                 ▼
┌────────────────────────────────────────────────────────────┐
│ Eval gates — src/evaluate.py                               │
│ retrieval accuracy · groundedness · extraction field F1    │
│ safety: no hallucinated medications, no invented follow-up │
│ or contact when gold is null                               │
└────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│ Fine-tuning — src/finetune.py (LoRA, distilgpt2)           │
│ Trains adapter weights on the synthetic train split to     │
│ emit the same fixed schema. Separate from the gated        │
│ baseline; see "Fine-tuning: what ran" below.               │
└────────────────────────────────────────────────────────────┘
```

Extraction schema (fixed, for every note):

```json
{
  "medications": [{"name": "lisinopril", "dose": "10 mg", "frequency": "daily"}],
  "follow_up_days": 7,
  "follow_up_clinic": "cardiology clinic",
  "red_flags": ["chest pain"],
  "contact": "555-0109"
}
```

Missing information is `null` (or `[]` for lists). Never filled in.

## How to run locally (zero paid APIs)

```bash
# 1. Measured baseline — no install needed
python3 src/evaluate.py
python3 -m unittest discover tests

# 2. Streamlit demo
pip install -r requirements.txt
streamlit run app.py

# 3. Optional Ollama path (local model, already tested here)
#    Start Ollama with the llama3.2:1b model, then use src/generate.py

# 4. Fine-tuning (LoRA) — see "Fine-tuning: what ran" below
pip install -r requirements-finetune.txt
python src/finetune.py --smoke-test --output models/lora-smoke
python src/finetune.py --epochs 3 --output models/lora-extraction
```

## Fine-tuning: what ran

Fine-tuning here is real parameter training (LoRA adapter weights on
distilgpt2, base model frozen), **not** few-shot prompting. The training
script is `src/finetune.py`; the training set is the 8 synthetic train
notes in `data/discharge_notes/notes.json`, each paired with its gold JSON.

> **What actually ran on this VM (Nov 4, 2026):**
> - The full pretrained LoRA run (`python src/finetune.py`) did **not**
>   complete here: the distilgpt2 weights download stalled at 0 bytes
>   (Hugging Face large-file transfer via this VM's proxy, ~40 minutes,
>   plus only ~240 MB free RAM at the time). No pretrained fine-tuned
>   model and no fine-tuned extraction metrics are claimed.
> - A small-scale, clearly-labelled smoke test **did** run and is verified:
>   a random-initialised distilgpt2 config with LoRA (r=8) — forward pass,
>   backward pass, and adapter save, on one synthetic train note.
>   Trainable parameters: **147,456 of 82,060,032 (0.1797%)**; initial
>   loss **11.0176**; adapter saved locally to
>   `models/lora-smoke-random-init/` (git-ignored, not pushed, and **not**
>   used for any extraction metric, because a random-init model is not a
>   trained extractor).
> - To run the real pretrained fine-tune, use a machine with Hugging Face
>   access (e.g. Srija's own computer): `pip install -r
>   requirements-finetune.txt` then `python src/finetune.py --epochs 3
>   --output models/lora-extraction`, then evaluate the adapter's output
>   against the test notes before claiming any fine-tuned number.

## Skills demonstrated — mapped to AI/ML Engineer and Data Scientist roles

| Skill | Where it shows in this project |
| --- | --- |
| RAG (retrieval + grounding) | TF-IDF retriever over a guidance corpus, extractive answers only, groundedness measured per key fact |
| Fine-tuning (parameter-efficient) | LoRA training script for structured extraction, trainable-parameter reporting, smoke-test mode |
| Evaluation design | Fixed test set, pre-set gates, field-level F1, retrieval accuracy, groundedness — not a single headline metric |
| Python | Clean, documented modules (`src/`), tests, reproducible eval script |
| Local LLM work | Ollama `llama3.2:1b` extraction path, temperature 0, JSON output, safety filtering of model output |
| Safety gating / responsible AI | Hallucination checks, null-instead-of-guess rule, documented safety flaw found and fixed, research-only disclaimer |

## Data: synthetic only, and the Kaggle decision

All discharge notes in this project are **synthetic examples written for this
demo** (IDs `SYN-001`–`SYN-012`, fake `555-` contacts, no patient names).
The guidance documents are short educational summaries written for this demo
from general public themes (heart-failure and diabetes follow-up, medication
reconciliation, care transitions) — they are **not** clinical guidelines and
not a substitute for them.

**Kaggle datasets were considered and deliberately not used**, because none
fit this task cleanly:

- **NY SPARCS de-identified inpatient discharges (Kaggle)** — genuinely
  de-identified and public, but structured discharge-level rows with no
  free-text notes and no medication/follow-up extraction schema; also a
  925 MB 2015 file. Wrong shape for this task.
- **MultiCaRe (Kaggle, CC BY-NC-SA)** — de-identified PubMed Central case
  reports, but case reports, not discharge notes, with no gold
  medication/follow-up handoff labels.
- **NBME patient-note scoring (Kaggle competition)** — data available only
  on request, not an open download. Excluded.
- **MIMIC copies on Kaggle** — license listed as Unknown, and MIMIC itself
  requires credentialed access. Excluded on principle.
- **Synthetic PII datasets on Kaggle** — contain intentionally fake
  identifiers (names, emails, ID numbers). Wrong lesson for a safety-gated
  project, and not discharge notes.

Forcing any of these in would have weakened the project's honesty. The
synthetic notes, with a fixed schema and gold labels, are the right data for
a *demonstration* of extraction + safety gating, and the README says so
plainly instead of implying real-data validation.

## Limitations — stated, not buried

1. **Synthetic, structured notes.** Real discharge notes are messier, less
   consistent, and full of abbreviations. The 1.000 extraction F1 will not
   transfer directly.
2. **Small test set.** 10 retrieval questions and 4 extraction test notes.
   Enough to prove the pipeline and gates; not enough to claim general
   performance.
3. **Guidance summaries, not guidelines.** The retrieval corpus is demo
   educational text, not authoritative clinical guidance.
4. **Small-model fine-tuning.** distilgpt2 + LoRA on 8 examples is a
   proof-of-training-loop, not a production extractor. A real version needs
   far more labelled notes and clinician validation.
5. **No clinical validation.** Nothing here has been reviewed by a
   clinician, and nothing here should be used for a real patient.

## What this project does NOT claim

No HIPAA compliance. No clinical experience. No real patient data. No
diagnosis or treatment recommendation. No paid APIs. No copied work from
any other person or project.
