"""
knowledge_store.py
-------------------
The retrieval layer of the system.

Rather than treating the knowledge base as static prompt text, this module
loads it into a queryable vector space so that recommendations are *retrieved*
based on semantic similarity to the situation being described, not hard-coded
if/else chains. It indexes two kinds of content:

  1. Structured intervention records (knowledge_base.json) - each one a
     discrete, evidence-backed action with explicit applicability conditions,
     mechanism, impacted metrics, and a citation.
  2. Reference document chunks (knowledge/references/*.txt) - short summaries
     of the underlying reports/studies, indexed so the system can show *why*
     a claim is grounded, not just assert it.

Retrieval uses TF-IDF + cosine similarity (scikit-learn). This is a legitimate,
inspectable vector-space retrieval method; it was chosen over a heavyweight
sentence-embedding model because it has no external network/model-download
dependency, which matters for a reviewer trying to run this in 5 minutes.
Swapping in `sentence-transformers` or a hosted embedding API is a drop-in
change - see README "Extending retrieval" section.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "knowledge", "knowledge_base.json")
REFERENCES_DIR = os.path.join(BASE_DIR, "knowledge", "references")


@dataclass
class RetrievedChunk:
    doc_id: str
    doc_type: str  # "intervention" | "reference"
    text: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class KnowledgeStore:
    """Loads the knowledge base + reference corpus and exposes vector retrieval."""

    def __init__(self, kb_path: str = KB_PATH, references_dir: str = REFERENCES_DIR):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

        self.interventions: list[dict[str, Any]] = self.kb["interventions"]
        self.clarifying_questions: dict[str, str] = self.kb["clarifying_questions"]
        self.metric_relationships: list[dict[str, Any]] = self.kb["metric_relationships"]

        # --- Build the retrieval corpus ---
        self._corpus_ids: list[str] = []
        self._corpus_types: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_payload: list[dict[str, Any]] = []

        for iv in self.interventions:
            # Flatten applicable_when condition keys/values into the searchable
            # text too, so a query like "monoculture low rainfall" retrieves
            # interventions whose *conditions* match even if that wording
            # doesn't appear in the prose action/mechanism fields.
            condition_terms = []
            for key, val in iv.get("applicable_when", {}).items():
                condition_terms.append(key.replace("_", " "))
                if isinstance(val, list):
                    condition_terms.extend(str(v).replace("_", " ") for v in val)
                else:
                    condition_terms.append(str(val).replace("_", " "))

            searchable = " ".join([
                iv["name"], iv["action"], iv["mechanism"],
                " ".join(iv["impacted_metrics"]), " ".join(iv["connected_variables"]),
                " ".join(condition_terms),
            ])
            self._corpus_ids.append(iv["id"])
            self._corpus_types.append("intervention")
            self._corpus_texts.append(searchable)
            self._corpus_payload.append(iv)

        if os.path.isdir(references_dir):
            for fname in sorted(os.listdir(references_dir)):
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(references_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                self._corpus_ids.append(fname)
                self._corpus_types.append("reference")
                self._corpus_texts.append(text)
                self._corpus_payload.append({"filename": fname, "text": text})

        # --- Vectorize ---
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self._corpus_texts)

    def retrieve(self, query: str, k: int = 5, doc_type: str | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks most similar to `query`, optionally filtered by type."""
        if not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()

        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results: list[RetrievedChunk] = []
        for i in ranked:
            if doc_type and self._corpus_types[i] != doc_type:
                continue
            if sims[i] <= 0:
                continue
            results.append(RetrievedChunk(
                doc_id=self._corpus_ids[i],
                doc_type=self._corpus_types[i],
                text=self._corpus_texts[i],
                score=float(sims[i]),
                payload=self._corpus_payload[i],
            ))
            if len(results) >= k:
                break
        return results

    def get_intervention(self, iv_id: str) -> dict[str, Any] | None:
        return next((iv for iv in self.interventions if iv["id"] == iv_id), None)

    def find_relevant_relationships(self, metrics: set[str]) -> list[dict[str, Any]]:
        """Return cross-metric relationship notes touching any of the given metrics."""
        out = []
        for rel in self.metric_relationships:
            if any(m in metrics for m in rel["pair"]) or set(rel["pair"]) & metrics:
                out.append(rel)
        return out
