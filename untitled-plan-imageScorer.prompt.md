## Plan: Image Scorer Overhaul

TL;DR — Convert to 0.0‑1.0 scoring, canonicalize folder organization (one-decimal base folders with dynamic subfolders when a base exceeds threshold), persist comparison history both in DB and per-image JSON, reconcile inconsistent code paths, and finish frontend (Ranking + Gallery) and API wiring. Implementation stays inside `custom_nodes/comfyui-image-scorer` and `output/` only.

**Steps**
1. Consolidate folder scheme (design & code) — choose canonical layout and implement deterministic folder computation.
2. Update path & movement logic — `path_handler`, `file_mover`, `folder_organizer`, and `image_processor` to use canonical layout and dynamic subfoldering when folder size > threshold.
3. Ensure initialization & migration — scanner runs using config image_root by default; migration script uses ImageProcessor to populate DB and move files safely.
4. Persist comparison history everywhere — make `comparisons` table returns insert id; append comparison entries to per-image JSON (id, other, winner, weight, timestamp, transitive_depth) using atomic writes.
5. Implement transitive inference & skipping rules — complete `transitive_inference.py` and integrate into `merge_sort_ranker.select_pair_for_comparison` (configurable depth & min weight).
6. Fix DB/API contract gaps — make `add_comparison` return id, ensure `record_comparison` updates DB, JSON, and moves files if tier changes.
7. Frontend finish & cleanup — fix `gallery.js` legacy code, make UI tabs explicit (Ranking, Gallery), ensure images load via `/output/ranked/<filename>` and show filters/sorting.
8. Tests & verification — unit + integration tests for file movement, DB writes, API endpoints, and UI rendering.
9. Documentation & config — update `SETUP_GUIDE.md`, add config keys: `subfolder_threshold`, `transitive_depth`, `use_fullpath_as_id`.

**Relevant files**
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/server.py — Flask app entry
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/image_processor.py — import & migration logic
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/file_management/path_handler.py — compute destinations
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/file_management/file_mover.py — atomic movement
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/file_management/folder_organizer.py — tier/folder creation
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/database/schema.py — DB init
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/database/images_table.py
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/database/comparisons_table.py
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/algorithm/merge_sort_ranker.py
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/algorithm/transitive_inference.py
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/algorithm/confidence_tracker.py
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/api/ranking_api_v2.py
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/api/gallery_api.py
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/frontend/html/index.html
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/frontend/html/gallery.html
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/frontend/js/app.js
- custom_nodes/comfyui-image-scorer/external_modules/step01ranking_new/frontend/js/gallery.js
- custom_nodes/comfyui-image-scorer/config/ranking_config.json

**Verification**
1. Run unit tests: `python external_modules/step01ranking_new/server.py --test`
2. Start server with scanner and migration: `python external_modules/step01ranking_new/server.py --image-root E:/ComfyUI/output`
3. Check API: `GET /api/v2/ranking/status`, `GET /api/v2/ranking/next-pair`, `POST /api/v2/ranking/submit-comparison`
4. In browser: visit `http://localhost:5001/` and verify Ranking tab shows pair images and Gallery tab lists images with filters and thumbnails.
5. Verify files moved into `output/scored/scored_X.X/` and that if a scored folder surpasses `subfolder_threshold` it created nested subfolders for second decimal.
6. Confirm `output/ranking_v2.db` contains `images` and `comparisons` rows; per-image JSON has `comparison_history` entries matching DB ids.

**Decisions & Assumptions**
- DB primary key currently is filename only. This plan assumes filenames are unique across the scanned image_root. If duplicates exist, we must choose one: (A) fail-loud (current), (B) use relative path as DB key, (C) auto-rename at move time. This is configurable — see questions.
- Folder naming canonicalized to `scored_{one_decimal}` base folders; subfolders created using second decimal when size threshold reached.
- Server should start background scanner from config by default (unless `--image-root` provided to override).
- Comparison history stored twice: normalized DB table for queries and appended to per-image JSON for portability.

**Immediate code inconsistencies found (to fix first)**
- `file_mover.check_duplicate_filename` scans `tier_* / sub_*` folders but other code uses `scored_*` folders — reconcile.
- `path_handler.compute_path_from_filename` does not implement dynamic subfoldering.
- `folder_organizer.get_tier_stats` and `file_mover` reference different folder naming patterns.
- `server.init_ranking_system` only starts scanner when `--image-root` provided — change to use config default.
- `gallery.js` contains legacy code paths (`/api/scores`, `/image/`, `/thumbnail/`) mixed with v2 API code — needs cleanup.
- `comparisons_table.add_comparison` returns bool only; return `lastrowid` for JSON sync.

**Risks & Mitigations**
- Duplicate filenames: risk of overwriting/misindexing — mitigation: detect duplicates at migration and stop or append suffixes per policy.
- Large migrations: scanning 30k+ images can take time — mitigation: batch processing and progress logging; use `--image-root` to run manual migration with monitoring.
- Concurrency/SQLite locks: keep writes serialized on single thread; use retry/backoff for DB writes.

**Next actions (I will do next if you approve)**
1. Ask required clarifying questions (I will present them now).  
2. After your answers, produce a step-by-step implementation checklist mapped to code edits and tests, and estimate time per step.

**Questions (please answer to finalize the plan)**
- How should duplicate filenames be handled? Options: `Fail-loud (current)`, `Use full relative path as DB key (recommended)`, `Auto-rename duplicates on move`.
- Confirm subfolder creation threshold (default `1000` images)? If different, provide number.
- Subfolder naming preference: `Nested two-decimal folders (scored_0.6/scored_0.65)` (recommended), `sub_NN numeric (scored_0.6/sub_05)`, or `flat two-decimal folders (scored_0.65)`.
- Maximum transitive inference depth? Options: `1`, `2` (recommended), `3`.
- Should server auto-start background scanner using `config["image_root"]` when `--image-root` omitted? Yes/No.
- Should per-image JSON store the DB comparison id for each history entry? Yes/No (recommended Yes).
- UI tab naming: prefer `Ranking` or `Compare` for first tab? (Ranking recommended)
- Migration behaviour: run automatically on server start, or require manual `--image-root` migration run? (Manual recommended for safety)

---

(Plan saved to session memory at `/memories/session/plan.md`.)