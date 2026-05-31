"""Metadata helpers for data-transform.

Both helpers in this file are currently unused in the repository. They are
kept commented out so the module stays importable while following the todo
instruction to comment out unused functions.
"""

# def write_error_log(error_log: list[dict[str, str]], path: str) -> None:
#     if error_log:
#         atomic_write_json(path, error_log, indent=2)
#     elif os.path.exists(path):
#         os.remove(path)

# def load_metadata_entry(meta_path: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
#     payload, ts, err = load_single_entry_mapping(meta_path)
#     if err:
#         if err == "not_found":
#             return None, None, "json_not_found"
#         if err.startswith("load_error"):
#             return None, None, "bad_json"
#         return None, None, err
#     return payload, ts, None

