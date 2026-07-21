"""Pair-type labels and tunable constants for the ranking algorithm."""

PAIR_TYPE_BOOTSTRAP = "bootstrap_seed"
PAIR_TYPE_INSERTION = "anchor_insert"
PAIR_TYPE_REFINEMENT = "uncertainty_refine"
PAIR_TYPE_FALLBACK = "fallback"
PAIR_TYPE_COLLAPSIBLE = "collapsible"

IMAGES_CACHE_TTL = 10
MAX_PAIR_CANDIDATES = 100

# Chain-merge phase keeps merging until fewer than this many chains remain.
MIN_CHAIN_THRESHOLD = 20
