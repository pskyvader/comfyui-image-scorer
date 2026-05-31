"""Data manager helpers.

The `collect_files` helper is currently unused in the repository, so it remains
commented out per the todo instructions.
"""

# from __future__ import annotations
#
# import os
# from collections.abc import Iterator
#
#
# def collect_files(root: str) -> Iterator[tuple[str, str]]:
#     for dirpath, _, files in os.walk(root):
#         for filename in files:
#             if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
#                 full = os.path.join(dirpath, filename)
#                 json_path = full.rsplit(".", 1)[0] + ".json"
#                 if os.path.exists(json_path):
#                     yield full, json_path

