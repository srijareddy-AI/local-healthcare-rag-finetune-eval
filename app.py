"""Streamlit demo for the care-transition RAG + extraction project.

Run locally:  streamlit run app.py
Requires:     pip install -r requirements.txt
Demo only, synthetic data only, not medical advice.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from extract import extract
from retrieve import TfidfRetriever, extractive_answer, load_chunks

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Care-Transition RAG (Demo)")
st.title("Care-Transition RAG — Synthetic Demo")
st.warning(
    "Research/demo only. Synthetic, de-identified notes only. "
    "Not medical advice. Human clinician review required before any real use."
)

notes = json.loads((ROOT / "data" / "discharge_notes" / "notes.json").read_text())
choice = st.selectbox("Choose a synthetic discharge note", [n["id"] for n in notes])
note = next(n for n in notes if n["id"] == choice)
st.text_area("Synthetic discharge note", note["note"], height=140)

if st.button("Extract structured handoff"):
    st.json(extract(note["note"]))

st.subheader("Ask the guidance corpus")
question = st.text_input("Question", "How soon after a heart failure hospital stay is follow-up usually arranged?")
if st.button("Retrieve answer"):
    retriever = TfidfRetriever(load_chunks(ROOT / "data" / "guidance"))
    retrieved = retriever.retrieve(question, top_k=3)
    st.write(extractive_answer(retrieved))
    st.write("Sources:", [r["source"] for r in retrieved])
