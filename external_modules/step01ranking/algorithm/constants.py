"""Pair-type labels and tunable constants for the ranking algorithm."""

# Pair type labels for debug/frontend
PAIR_TYPE_COLLAPSIBLE = "collapsible"
PAIR_TYPE_WORST_WITH_WORST = "worst_with_worst"
PAIR_TYPE_UNCONNECTED = "unconnected"
PAIR_TYPE_REFINEMENT = "refinement"
PAIR_TYPE_FALLBACK = "fallback"

# Cache TTL for the all-images list (seconds)
IMAGES_CACHE_TTL = 5.5

# Maximum candidate images for fallback pair selection
MAX_PAIR_CANDIDATES = 100
