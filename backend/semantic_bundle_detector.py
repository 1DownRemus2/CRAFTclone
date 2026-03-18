# semantic_bundle_detector.py
import numpy as np
from typing import Tuple, Optional
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    BUNDLE_THRESHOLD,
    TARGETED_THRESHOLD,
    DOMAIN_CONFIDENCE_WEIGHT,
    BUNDLE_DOMAIN_CONFIDENCE_THRESHOLD,
    SHORT_QUERY_WORD_THRESHOLD,
)
from embedding_utils import l2_normalize


# ---------------------------------------------------------------------------
# Training examples — edit here or move to a YAML file for tuning without
# touching logic code.
# ---------------------------------------------------------------------------

BUNDLE_EXAMPLES = [
    "build a complete system",
    "set up everything",
    "create a full solution",
    "install the entire setup",
    "make a comprehensive system",
    "configure the entire system",
    "establish a complete solution",
    "assemble everything together",
    "put together a full solution",
    "construct the whole system",
    "complete smart home",
    "full home office",
    "entire workspace",
    "whole system setup",
    "complete office setup",
    "I want everything",
    "need the full package",
    "want a complete setup",
    "need everything configured",
    "want the whole thing",
    "comprehensive solution",
    "all components together",
    "full integration",
    "complete package",
    "entire ecosystem",
]

TARGETED_EXAMPLES = [
    "I only want one thing",
    "just need a specific item",
    "only this device",
    "just this one",
    "only looking for one item",
    "just want this particular item",
    "only interested in this",
    "exclusively need this device",
    "solely interested in this",
    "merely looking for one",
    "I need a monitor",
    "I need a desk",
    "I need a sensor",
    "need one device",
    "looking for a specific device",
    "just the monitor",
    "only the desk",
    "specifically a sensor",
    "one device only",
    "a single item",
    "all I need is one thing",
    "that's all I need",
    "nothing else needed",
    "just looking for one",
    "one specific thing",
]


class SemanticBundleIntentDetector:
    """
    Classifies a user query as either "bundle" (wants a full system) or
    "targeted" (wants one specific capability/device).

    Optimisations vs original:
    - Accepts a pre-loaded shared model.
    - All thresholds come from config.py.
    - Training example lists moved to module level (easier to edit/test).
    - explain_scores reuses the query vector from compute_scores instead of
      re-encoding the query string a second time.
    """

    def __init__(
        self,
        model: Optional[SentenceTransformer] = None,
        bundle_threshold: float = BUNDLE_THRESHOLD,
        targeted_threshold: float = TARGETED_THRESHOLD,
        domain_confidence_weight: float = DOMAIN_CONFIDENCE_WEIGHT,
        domain_confidence_threshold: float = BUNDLE_DOMAIN_CONFIDENCE_THRESHOLD,
        short_query_threshold: int = SHORT_QUERY_WORD_THRESHOLD,
    ):
        self.model = model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.bundle_threshold = bundle_threshold
        self.targeted_threshold = targeted_threshold
        self.domain_confidence_weight = domain_confidence_weight
        self.domain_confidence_threshold = domain_confidence_threshold
        self.short_query_threshold = short_query_threshold

        # Pre-encode and normalise training examples once at init
        self.bundle_embeddings: np.ndarray = l2_normalize(
            self.model.encode(BUNDLE_EXAMPLES)
        )
        self.targeted_embeddings: np.ndarray = l2_normalize(
            self.model.encode(TARGETED_EXAMPLES)
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode and L2-normalise a query string."""
        return l2_normalize(self.model.encode([query])[0])

    def _max_sim(self, query_vec: np.ndarray, embeddings: np.ndarray) -> float:
        """Max cosine similarity between a normalised query vec and a matrix."""
        return float(np.max(embeddings @ query_vec))

    def _is_short_query(self, query: str) -> bool:
        return len(query.split()) <= self.short_query_threshold

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compute_scores(self, query: str) -> Tuple[np.ndarray, float, float]:
        """
        Encode the query and compute bundle + targeted similarity scores.

        Returns:
            (query_vec, bundle_score, targeted_score)
            Returning query_vec avoids re-encoding in callers like explain_scores.
        """
        query_vec = self._encode_query(query)
        return (
            query_vec,
            self._max_sim(query_vec, self.bundle_embeddings),
            self._max_sim(query_vec, self.targeted_embeddings),
        )

    def detect(
        self,
        query: str,
        domain: str,
        domain_confidence: float,
    ) -> Tuple[bool, str]:
        """
        Classify the query as bundle or targeted intent.

        Args:
            query:             Raw user query string.
            domain:            Domain detected by IntentEngine ("unknown" if none).
            domain_confidence: Confidence score from IntentEngine.

        Returns:
            (is_bundle, reason) — reason is a human-readable explanation string.
        """
        query_vec, bundle_score, targeted_score = self.compute_scores(query)
        is_short = self._is_short_query(query)

        # 1. Strong targeted signal overrides everything
        if targeted_score >= self.targeted_threshold:
            return False, (
                f"Strong targeted intent detected "
                f"(targeted_sim={targeted_score:.3f} >= {self.targeted_threshold:.3f})"
            )

        # 2. Strong bundle signal
        if bundle_score >= self.bundle_threshold:
            return True, (
                f"Strong bundle intent detected "
                f"(bundle_sim={bundle_score:.3f} >= {self.bundle_threshold:.3f})"
            )

        # 3. Short query — lean on domain confidence
        if is_short and domain != "unknown":
            if domain_confidence >= self.domain_confidence_threshold:
                return True, (
                    f"Short query with high domain confidence "
                    f"(words={len(query.split())}, "
                    f"domain_conf={domain_confidence:.3f} >= {self.domain_confidence_threshold:.3f})"
                )
            if domain_confidence >= 0.55:
                weighted = (
                    bundle_score * (1 - self.domain_confidence_weight)
                    + domain_confidence * self.domain_confidence_weight
                )
                if weighted >= self.bundle_threshold:
                    return True, (
                        f"Short query with medium domain confidence "
                        f"(bundle_sim={bundle_score:.3f}, "
                        f"domain_conf={domain_confidence:.3f}, "
                        f"weighted={weighted:.3f})"
                    )

        # 4. Longer query — domain confidence as tiebreaker
        if (
            not is_short
            and domain != "unknown"
            and domain_confidence >= self.domain_confidence_threshold
        ):
            weighted = (
                bundle_score * (1 - self.domain_confidence_weight)
                + domain_confidence * self.domain_confidence_weight
            )
            if weighted >= self.bundle_threshold:
                return True, (
                    f"Bundle via domain confidence "
                    f"(bundle_sim={bundle_score:.3f}, "
                    f"domain_conf={domain_confidence:.3f}, "
                    f"weighted={weighted:.3f})"
                )

        # 5. Direct score comparison with margin to avoid flip-flopping
        margin = 0.05
        if bundle_score > targeted_score + margin:
            return True, (
                f"Bundle score significantly higher "
                f"(bundle={bundle_score:.3f} > targeted={targeted_score:.3f} + {margin})"
            )
        if targeted_score > bundle_score + margin:
            return False, (
                f"Targeted score significantly higher "
                f"(targeted={targeted_score:.3f} > bundle={bundle_score:.3f} + {margin})"
            )

        # 6. Ambiguous — default to targeted (safer)
        return False, (
            f"Ambiguous scores, defaulting to targeted "
            f"(bundle={bundle_score:.3f}, targeted={targeted_score:.3f})"
        )

    def explain_scores(self, query: str) -> str:
        """Detailed debugging explanation of bundle/targeted scores."""
        # Reuse compute_scores so the query is only encoded once
        query_vec, bundle_score, targeted_score = self.compute_scores(query)

        bundle_sims = self.bundle_embeddings @ query_vec
        targeted_sims = self.targeted_embeddings @ query_vec

        top3_bundle = np.argsort(bundle_sims)[-3:][::-1]
        top3_targeted = np.argsort(targeted_sims)[-3:][::-1]

        lines = [
            f'Query: "{query}"',
            f"Word count: {len(query.split())} (short: {self._is_short_query(query)})",
            "",
            f"Bundle similarity: {bundle_score:.3f}",
            "  Top matches:",
        ]
        for idx in top3_bundle:
            lines.append(f'    - "{BUNDLE_EXAMPLES[idx]}" ({bundle_sims[idx]:.3f})')

        lines += ["", f"Targeted similarity: {targeted_score:.3f}", "  Top matches:"]
        for idx in top3_targeted:
            lines.append(f'    - "{TARGETED_EXAMPLES[idx]}" ({targeted_sims[idx]:.3f})')

        return "\n".join(lines)