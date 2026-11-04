"""Optional local Ollama generation path (llama3.2:1b).

Uses the Ollama HTTP API only. If Ollama is not running, this module
raises a clear error instead of falling back to a paid API or inventing
an answer. Generation is always checked against retrieved text: any
medication in the output that is not in the source note is stripped.
"""
import json
import os
import re

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11436")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


def ollama_extract(note, model=MODEL):
    prompt = (
        "Extract structured data from this SYNTHETIC discharge note. "
        "Return only a JSON object with keys medications (list of "
        "{name, dose, frequency}), follow_up_days (integer or null), "
        "follow_up_clinic (string or null), red_flags (list), contact "
        "(string or null). If a field is not present, use null or []. "
        "Do not invent any value.\n\n"
        f"Note: {note}\n\nJSON:"
    )
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
            "format": "json",
        },
        timeout=120,
    )
    response.raise_for_status()
    text = response.json()["message"]["content"]
    data = json.loads(text)
    # Safety: drop any medication not literally in the note, and drop
    # placeholder entries such as "none" (from "none documented") that
    # a small model may emit instead of an empty list.
    lower = note.lower()
    cleaned = []
    for med in data.get("medications", []):
        name = (med.get("name") or "").lower().strip()
        if name in ("", "none", "not documented", "null"):
            continue
        if name in lower and med.get("dose") not in (None, "null"):
            cleaned.append(med)
    data["medications"] = cleaned
    if data.get("red_flags") is None:
        data["red_flags"] = []
    for key in ("follow_up_days", "follow_up_clinic", "contact"):
        data.setdefault(key, None)
    data.setdefault("red_flags", [])
    return data
