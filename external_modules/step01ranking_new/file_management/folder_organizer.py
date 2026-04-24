"""Folder organizer - maintain score folder structure."""


def ensure_tier_structure() -> bool:
    """
    Ensure score folders exist (scored_0.0 through scored_1.0).
    Called once during initialization.

    Returns:
        True if successful
    """
    try:
        # Import here to avoid circular imports
        from .path_handler import get_ranked_root

        ranked_root = get_ranked_root()
        ranked_root.mkdir(parents=True, exist_ok=True)

        # Create scored_X.X folders for common score values
        # Users can have images at any score, but we pre-create common ones
        for i in range(11):
            score = i / 10.0
            score_folder = ranked_root / f"scored_{score:.1f}"
            score_folder.mkdir(parents=True, exist_ok=True)

        return True
    except Exception as e:
        print(f"Error creating score structure: {e}")
        return False
