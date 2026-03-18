# capability_intent_engine.py
import numpy as np
from typing import Set, Dict, Optional
from sentence_transformers import SentenceTransformer

from config import CAPABILITY_INTENT_THRESHOLD, EMBEDDING_MODEL_NAME
from domains import DOMAINS
from embedding_utils import build_capability_embeddings, l2_normalize, max_cosine_similarity


class CapabilityIntentEngine:
    """
    Detects explicit capability mentions in user queries using semantic
    similarity against capability descriptions from DOMAINS.

    Optimisations vs original:
    - Accepts a pre-loaded shared model (no redundant loading).
    - Capability embeddings are pre-normalised at init.
    - detect_with_scores reuses the query vector computed inside detect()
      instead of re-encoding the query a second time.
    """

    def __init__(self, model: Optional[SentenceTransformer] = None):
        self.model = model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._build_capability_embeddings()

    def _build_capability_embeddings(self) -> None:
        """Build and L2-normalise capability embeddings from DOMAINS."""
        raw = build_capability_embeddings(self.model, DOMAINS)
        self.capability_embeddings: Dict[str, np.ndarray] = {
            cap: l2_normalize(embs) for cap, embs in raw.items()
        }

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode and normalise a query string into a unit vector."""
        return l2_normalize(self.model.encode([query])[0])

    def detect(self, query: str) -> Set[str]:
        """
        Detect which capabilities are explicitly mentioned in the query.

        Returns:
            Set of capability names whose max similarity meets the threshold.
        """
        query_vec = self._encode_query(query)
        focused: Set[str] = set()

        for cap_name, cap_embs in self.capability_embeddings.items():
            sim = max_cosine_similarity(query_vec, cap_embs)
            if sim >= CAPABILITY_INTENT_THRESHOLD:
                focused.add(cap_name)
                print(
                    f"  → Detected capability intent: '{cap_name}' "
                    f"(similarity: {sim:.3f})"
                )

        return focused

    def detect_with_scores(self, query: str) -> Dict[str, float]:
        """
        Return all capabilities with their similarity scores (for debugging).

        The query is encoded once and reused across all capability comparisons.
        """
        query_vec = self._encode_query(query)
        scores = {
            cap_name: max_cosine_similarity(query_vec, cap_embs)
            for cap_name, cap_embs in self.capability_embeddings.items()
        }
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))