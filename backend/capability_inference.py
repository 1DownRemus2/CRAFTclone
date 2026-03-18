# capability_inference.py
import numpy as np
from typing import Dict, Set, List, Optional
from sentence_transformers import SentenceTransformer

from config import CAPABILITY_INFERENCE_THRESHOLD, EMBEDDING_MODEL_NAME
from domains import DOMAINS
from embedding_utils import build_capability_embeddings, l2_normalize, max_cosine_similarity


class CapabilityInferenceEngine:
    """
    Infers device capabilities using semantic similarity between device
    descriptions and capability definitions from DOMAINS.

    Optimisations vs original:
    - Accepts a pre-loaded shared model instead of loading its own copy.
    - Capability embeddings are pre-normalised once at init so similarity
      comparisons become cheap dot products.
    - enrich_device performs an explicit deep copy of 'provides'/'requires'
      sets to prevent mutation of the source DEVICE_REGISTRY.
    - Adds an 'originally_empty' flag so callers know which devices received
      purely inferred capabilities (replaces the hard-coded ID whitelist in
      app.py).
    """

    def __init__(self, model: Optional[SentenceTransformer] = None):
        self.model = model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._build_capability_embeddings()

    def _build_capability_embeddings(self) -> None:
        """
        Build and L2-normalise embeddings for every capability in DOMAINS.
        Normalising once here means infer_capabilities can use plain dot
        products instead of the full cosine formula on every comparison.
        """
        raw = build_capability_embeddings(self.model, DOMAINS)
        # Normalise each row so similarity == dot product
        self.capability_embeddings: Dict[str, np.ndarray] = {
            cap: l2_normalize(embs) for cap, embs in raw.items()
        }

    def infer_capabilities(self, device: Dict) -> Set[str]:
        """
        Return the full set of capabilities for a device (explicit + inferred).

        The device description and tags are concatenated, encoded, and compared
        against every known capability.  Any capability whose max similarity
        meets the threshold is included.
        """
        # Preserve explicitly declared capabilities
        inferred: Set[str] = set(device.get("provides", set()))

        device_text = device["description"] + " " + " ".join(device.get("tags", []))
        # Normalise the device vector once; reuse across all capability comparisons
        device_vec = l2_normalize(self.model.encode([device_text])[0])

        for cap_name, cap_embs in self.capability_embeddings.items():
            sim = max_cosine_similarity(device_vec, cap_embs)
            if sim >= CAPABILITY_INFERENCE_THRESHOLD:
                inferred.add(cap_name)
                if device.get("originally_empty"):
                    print(
                        f"  → Inferred '{cap_name}' for {device['id']} "
                        f"(similarity: {sim:.3f})"
                    )

        return inferred

    def enrich_device(self, device: Dict) -> Dict:
        """
        Return a new dict with inferred capabilities merged in.

        Uses explicit set copies for 'provides' and 'requires' so the
        original DEVICE_REGISTRY entries are never mutated.
        """
        originally_empty = not bool(device.get("provides"))
        enriched = {
            **device,
            "provides": set(device.get("provides", set())),   # safe copy
            "requires": set(device.get("requires", set())),   # safe copy
            "originally_empty": originally_empty,
            "inferred": True,
        }
        enriched["provides"] = self.infer_capabilities(enriched)
        return enriched

    def enrich_registry(self, registry: List[Dict]) -> List[Dict]:
        """Enrich every device in the registry and return a new list."""
        return [self.enrich_device(device) for device in registry]