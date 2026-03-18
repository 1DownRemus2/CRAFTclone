# intent_engine.py
import faiss
import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from config import DOMAIN_CONFIDENCE_THRESHOLD, EMBEDDING_MODEL_NAME
from domains import DOMAINS


class IntentEngine:
    """
    Detects user intent (domain) using a FAISS index of domain examples.

    Optimisations vs original:
    - Accepts a pre-loaded shared model.
    - Guards against an empty DOMAINS dict (detect_top_k no longer crashes
      with k=0 when the index is empty).
    """

    def __init__(self, model: Optional[SentenceTransformer] = None):
        self.model = model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._build_index()

    def _build_index(self) -> None:
        """Build a FAISS inner-product index from all domain intent examples."""
        self.sentences: List[str] = []
        self.labels: List[str] = []

        for domain, spec in DOMAINS.items():
            for example in spec["intent_examples"]:
                self.sentences.append(example)
                self.labels.append(domain)

        if not self.sentences:
            # Empty domains — index will hold nothing; detect() will always
            # return ("unknown", 0.0) which is the correct fallback.
            self.index = None
            return

        embeddings = self.model.encode(self.sentences).astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def detect(self, query: str) -> Tuple[str, float]:
        """
        Detect the domain from a user query.

        Returns:
            (domain_name, confidence) where domain_name is 'unknown' if
            confidence is below DOMAIN_CONFIDENCE_THRESHOLD.
        """
        if self.index is None or self.index.ntotal == 0:
            return "unknown", 0.0

        vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)

        similarities, indices = self.index.search(vec, k=1)
        idx = int(indices[0][0])
        confidence = float(similarities[0][0])

        if confidence < DOMAIN_CONFIDENCE_THRESHOLD:
            return "unknown", confidence

        return self.labels[idx], confidence

    def detect_top_k(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return the top-k domain matches with their confidence scores."""
        if self.index is None or self.index.ntotal == 0:
            return []

        vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)

        k_actual = min(k, self.index.ntotal)
        similarities, indices = self.index.search(vec, k=k_actual)

        return [
            (self.labels[int(indices[0][i])], float(similarities[0][i]))
            for i in range(k_actual)
        ]