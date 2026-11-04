"""Pure-stdlib TF-IDF retrieval over the public guidance summaries.

No external embedding service is used. Chunks are sentence groups from the
guidance documents, scored by TF-IDF cosine similarity to the question.
"""
import math
import re
from pathlib import Path

STOPWORDS = set(
    "a an the and or of to in is are was were be been for with on at by "
    "it its this that as from after before".split()
)


def tokenize(text):
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def load_chunks(guidance_dir):
    chunks = []
    for path in sorted(Path(guidance_dir).glob("*.md")):
        text = path.read_text()
        # Split into sentence groups of at most ~3 sentences.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        for i in range(0, len(sentences), 3):
            body = " ".join(sentences[i : i + 3])
            chunks.append({"source": path.name, "text": body})
    return chunks


class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [tokenize(c["text"]) for c in chunks]
        n = len(self.docs)
        df = {}
        for tokens in self.docs:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def _vector(self, tokens):
        vec = {}
        for token in tokens:
            vec[token] = vec.get(token, 0) + 1
        return {t: count * self.idf.get(t, 1.0) for t, count in vec.items()}

    @staticmethod
    def _cosine(a, b):
        dot = sum(a.get(t, 0) * b[t] for t in b)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def retrieve(self, query, top_k=3):
        qvec = self._vector(tokenize(query))
        scored = []
        for chunk, tokens in zip(self.chunks, self.docs):
            score = self._cosine(qvec, self._vector(tokens))
            scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{"score": round(score, 4), **chunk} for score, chunk in scored[:top_k]]


def extractive_answer(retrieved):
    """Return the top chunk verbatim. Never generate text beyond the source."""
    if not retrieved:
        return ""
    return retrieved[0]["text"]
