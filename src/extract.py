"""Structured extraction from synthetic discharge notes.

Rule-based baseline extractor. Safety rule, applied to every output:
missing information is returned as null (or an empty list for medications
and red flags), never filled in. Any medication not literally present in
the note text is dropped, so the extractor cannot hallucinate a drug.
"""
import json
import re
from pathlib import Path

MED_RE = re.compile(r"([a-z]+)\s+(\d+\s*mg)\s+(daily|twice daily)")
FOLLOWUP_RE = re.compile(r"Follow-up:\s+in (\d+) days with ([^.]+)\.")
CONTACT_RE = re.compile(r"Care team contact:\s*(\d{3}-\d{4})")


def _section(note, label, end_markers):
    pattern = rf"{label}:\s*(.+?)(?:\.\s+(?:{'|'.join(end_markers)})\b|\.$)"
    match = re.search(pattern, note)
    return match.group(1).strip() if match else None


def extract(note):
    lower = note.lower()
    meds_text = _section(note, "Medications", ["Follow-up"])
    medications = []
    if meds_text and "none documented" not in meds_text.lower():
        for name, dose, frequency in MED_RE.findall(meds_text.lower()):
            # Safety: the drug name must literally appear in the note.
            if name in lower:
                medications.append(
                    {"name": name, "dose": dose, "frequency": frequency}
                )
    followup = FOLLOWUP_RE.search(note)
    follow_up_days = int(followup.group(1)) if followup else None
    follow_up_clinic = followup.group(2).strip() if followup else None
    flags_text = _section(note, "Red flags", ["Care team contact"])
    red_flags = []
    if flags_text and "none documented" not in flags_text.lower():
        red_flags = [f.strip() for f in flags_text.split(",") if f.strip()]
    contact_match = CONTACT_RE.search(note)
    contact = contact_match.group(1) if contact_match else None
    return {
        "medications": medications,
        "follow_up_days": follow_up_days,
        "follow_up_clinic": follow_up_clinic,
        "red_flags": red_flags,
        "contact": contact,
    }


def load_notes(path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    import sys

    for item in load_notes(sys.argv[1]):
        print(item["id"], json.dumps(extract(item["note"])))
