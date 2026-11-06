"""Part D (retrieval): chunk the policy documents, search them with TF-IDF, answer from context.

Run with ``python -m src.policy_rag`` to evaluate retrieval and grounding with 10-word and
zero-word overlap. Results go to ``outputs/metrics/part_d_rag.json``.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm
from src.config import POLICIES, RAW, save_metrics

logger = logging.getLogger("copilot.rag")

CHUNK_WORDS = 40
OVERLAP_WORDS = 10
FALLBACK_PREFIX = "I could not find"
SYSTEM_PROMPT = ("POLICY ASSISTANT: answer the question using only the supplied context. If the context does not "
                 "contain the answer, say: I could not find that information in the policy documents.")


def chunk_words(words: list[str], size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[list[str]]:
    """Fixed-size word windows; stops after the first window that reaches the last word."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be positive and overlap must lie in [0, size)")
    chunks, start, step = [], 0, size - overlap
    while start < len(words):
        chunks.append(words[start:start + size])
        if start + size >= len(words):
            break
        start += step
    return chunks


class PolicyRAG:
    """Retrieval index over the policy documents, with grounded answering."""

    def __init__(self, chunk_size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> None:
        self.chunks: list[dict] = []
        paths = sorted(POLICIES.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(f"no policy documents in {POLICIES}; run generate_data.py first")
        for path in paths:
            words = path.read_text(encoding="utf-8").split()
            self.chunks += [{"source": path.name, "text": " ".join(c)} for c in chunk_words(words, chunk_size, overlap)]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Top ``k`` chunks by cosine similarity, as dictionaries with source, score and text."""
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = np.argsort(-scores, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]}
                for i in ranked]

    def answer(self, question: str) -> dict:
        """Retrieve three chunks and ask the model to answer from them alone."""
        hits = self.retrieve(question, 3)
        context = "\n".join(f"[{hit['source']}]\n{hit['text']}" for hit in hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        reply = mock_llm.chat(messages).choices[0].message.content
        return {"answer": reply, "sources": list(dict.fromkeys(hit["source"] for hit in hits)), "retrieved": hits}


@lru_cache(maxsize=1)
def default_rag() -> PolicyRAG:
    """The shared index (40-word chunks, 10-word overlap), built on first use."""
    return PolicyRAG()


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Search the shared index."""
    return default_rag().retrieve(query, k)


def answer(question: str) -> dict:
    """Answer from the shared index. Returns the text and the sources used."""
    return default_rag().answer(question)


def evaluate(rag: PolicyRAG) -> dict:
    """Score retrieval, fact grounding and the fallback on ``policy_eval.csv``."""
    questions = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)
    rows = []
    for row in questions.itertuples(index=False):
        result = rag.answer(row.question)
        answerable = row.expected_doc != "NONE"
        rows.append({"question": row.question, "answerable": answerable,
                     "doc_in_top3": bool(answerable and row.expected_doc in result["sources"]),
                     "fact_in_answer": bool(answerable and row.expected_fact in result["answer"]),
                     "fallback": result["answer"].startswith(FALLBACK_PREFIX), "answer": result["answer"]})
    frame = pd.DataFrame(rows)
    answerable, unanswerable = frame[frame["answerable"]], frame[~frame["answerable"]]
    return {"chunks": len(rag.chunks),
            "chunks_by_source": pd.Series([c["source"] for c in rag.chunks]).value_counts().to_dict(),
            "answerable_total": len(answerable), "doc_in_top3": int(answerable["doc_in_top3"].sum()),
            "fact_in_answer": int(answerable["fact_in_answer"].sum()),
            "unanswerable_total": len(unanswerable), "fallback": int(unanswerable["fallback"].sum()),
            "failed_questions": answerable[~(answerable["doc_in_top3"] & answerable["fact_in_answer"])][
                ["question", "doc_in_top3", "fact_in_answer", "answer"]].to_dict("records"),
            "per_question": frame.to_dict("records")}


def run() -> dict:
    """Evaluate the 10-word-overlap index and the zero-overlap index."""
    default = PolicyRAG()
    payload = {"overlap_10": evaluate(default), "overlap_0": evaluate(PolicyRAG(overlap=0)),
               "sample_query_top3_sources": [h["source"] for h in default.retrieve("refund policy late delivery")]}
    save_metrics("part_d_rag", payload)
    logger.info("Part D retrieval finished: %d chunks", payload["overlap_10"]["chunks"])
    return payload


if __name__ == "__main__":
    from src.config import setup_logging
    setup_logging()
    run()
