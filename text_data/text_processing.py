# Compatibility shim: expose top-level text_data.text_processing for tests and legacy imports
from .prepare.text_processing import (  # noqa: F401
    load_text_index,
    process_text_files,
    save_text_index,
)

__all__ = ["load_text_index", "process_text_files", "save_text_index"]
