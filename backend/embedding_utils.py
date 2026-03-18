# embedding_utils.py
# Shared utilities for building capability embeddings from DOMAINS.
# Both CapabilityIntentEngine and CapabilityInferenceEngine used identical
# loops — this module owns that logic once.

import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer


def build_capability_embeddings(
    model: SentenceTransformer,
    domains: Dict,
) -> Dict[str, np.ndarray]:
    """
    Build a {capability_name -> embeddings_array} map from a DOMAINS dict.

    Each capability may have multiple description phrases; each phrase gets
    its own embedding row so callers can compute max-similarity across them.

    Args:
        model:   A loaded SentenceTransformer instance (shared, not reloaded).
        domains: The DOMAINS dict from domains.py.

    Returns:
        Dict mapping capability name -> np.ndarray of shape (n_phrases, dim).
    """
    capability_map: Dict[str, list] = {}

    for domain_spec in domains.values():
        for cap_name, descriptions in domain_spec["goal_capabilities"].items():
            if cap_name not in capability_map:
                capability_map[cap_name] = []
            # Extend so cross-domain caps with the same name are merged
            capability_map[cap_name].extend(descriptions)

    return {
        cap_name: model.encode(descriptions)
        for cap_name, descriptions in capability_map.items()
    }


def max_cosine_similarity(
    query_vec: np.ndarray,
    matrix: np.ndarray,
) -> float:
    """
    Return the maximum cosine similarity between query_vec and any row in matrix.

    Both query_vec and matrix rows are assumed to be already L2-normalised.
    Using pre-normalised vectors turns cosine similarity into a plain dot product,
    avoiding repeated norm computation inside hot loops.
    """
    return float(np.max(matrix @ query_vec))


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of vec (works for both 1-D and 2-D arrays)."""
    if vec.ndim == 1:
        return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)