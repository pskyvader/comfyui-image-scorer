"""Compatibility shims for relocated text processing utilities.

The real implementations now live in text_data.text_processing. Importing and
re-exporting here keeps older imports working while centralizing logic in the
new text_data module.
"""

from text_data.text_processing import (  # noqa: F401
    load_text_index,
    process_text_files,
    save_text_index,
)

__all__ = [
    "load_text_index",
    "process_text_files",
    "save_text_index",
]
