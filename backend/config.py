# config.py
# Central configuration for all thresholds and tunable parameters.
# Edit values here instead of hunting across multiple files.

# ── Similarity thresholds ──────────────────────────────────────────────────────

# Minimum cosine similarity for a device description to be assigned a capability
CAPABILITY_INFERENCE_THRESHOLD: float = 0.5

# Minimum cosine similarity for a query to be treated as mentioning a capability
CAPABILITY_INTENT_THRESHOLD: float = 0.55

# Minimum cosine similarity for the domain detector to accept a domain match
DOMAIN_CONFIDENCE_THRESHOLD: float = 0.6

# ── Bundle detector thresholds ─────────────────────────────────────────────────

# Minimum bundle-example similarity to declare "bundle" intent
BUNDLE_THRESHOLD: float = 0.50

# Minimum targeted-example similarity to override bundle with "targeted" intent
TARGETED_THRESHOLD: float = 0.58

# Weight given to domain confidence when computing a weighted bundle score
DOMAIN_CONFIDENCE_WEIGHT: float = 0.35

# Minimum domain confidence before it influences the bundle decision
BUNDLE_DOMAIN_CONFIDENCE_THRESHOLD: float = 0.70

# Queries with this many words or fewer are treated as "short" for bundle logic
SHORT_QUERY_WORD_THRESHOLD: int = 3

# ── Model ──────────────────────────────────────────────────────────────────────

# SentenceTransformer model name used by all engines
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"