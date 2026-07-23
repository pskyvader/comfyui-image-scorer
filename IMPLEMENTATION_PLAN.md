# Implementation Plan: Architecture Restructure

**Constraint:** The README.md architecture documentation is **immutable** — it defines the target structure and rules. All implementation must conform to it exactly. No modifications to README structure or rules without explicit architectural decision.

**Non-Negotiable Rule:** After **every single phase**, the entire system must work end-to-end:
- All existing tests pass (`pytest`)
- Type checking passes (`pyright`)
- ComfyUI nodes load correctly (programmatic verification)
- Server starts without errors
- No import errors anywhere

If a phase breaks anything — **stop, fix, or roll back**. Do not proceed to the next phase.

---

## Phase 0: Preparation & Baseline

- [ ] Ensure `pyproject.toml` has correct package root (`comfyui_image_scorer/`)
- [ ] Verify `uv` and `pyright` work in ComfyUI venv
- [ ] **Run full test suite** — establish baseline: `pytest` (must pass 100%)
- [ ] **Run type check** — `pyright` (must pass 100%)
- [ ] **Verify ComfyUI node import** — run programmatic test from README
- [ ] **Start server** — verify no import errors
- [ ] Commit baseline: `git commit -m "Baseline before restructure"`

---

## Phase 1: Create New Package Skeleton

Create directory structure (empty, with minimal `__init__.py`):

```
comfyui_image_scorer/
├── core/
│   ├── configuration/
│   ├── filesystem/
│   ├── observability/
│   ├── io/
│   └── utilities/
├── domain/
│   ├── comparison/
│   ├── database/
│   ├── data_transformation/
│   ├── training/
│   ├── analysis/
│   ├── graph/
│   ├── vectors/
│   └── loading/
├── application/
│   ├── services/
│   ├── dto/
│   └── ports/
├── adapters/
│   ├── server/
│   │   ├── routing/
│   │   ├── endpoints/
│   │   └── middleware/
│   ├── comfyui/
│   │   ├── nodes/
│   │   │   ├── aesthetic_score/
│   │   │   ├── ranking/
│   │   │   ├── gallery/
│   │   │   └── maps/
│   │   ├── input_adapters/
│   │   ├── output_adapters/
│   │   └── node_registry.py
│   └── cli/
│       ├── commands/
│       │   ├── server.py
│       │   ├── training.py
│       │   ├── vectors.py
│       │   └── database.py
│       └── output.py
└── infrastructure/
    ├── persistence/
    ├── ml_models/
    └── external_services/
```

**Rule:** Each leaf directory gets an empty `__init__.py` (package marker only — no re-exports).

### Post-Phase 1 Validation (Must Pass)
```bash
pytest                    # All existing tests still pass
pyright                   # No new type errors
python -c "import comfyui_image_scorer"  # Package imports
```

---

## Phase 2: Move Core Utilities (Zero-Risk)

Move files that have **no external dependencies** (only stdlib + each other). **Move ONE file at a time**, validate after each.

| From | To |
|---|---|
| `shared/config.py` | `core/configuration/settings.py` |
| `shared/paths.py` | `core/filesystem/paths.py` |
| `shared/logger.py` | `core/observability/logger.py` |
| `shared/io.py` | `core/io/serialization.py` |
| `shared/utils.py` | `core/utilities/general.py` |
| `shared/helpers.py` | `core/utilities/helpers.py` |
| `shared/tasks.py` | `application/services/task_service.py` |

**For EACH file move:**
1. Create target file with updated absolute imports
2. Update all **other files** that import the moved file to use new path
3. Run validation:

```bash
# After EACH file move:
pytest                           # All tests pass
pyright                          # No type errors
python -c "
import sys; sys.path.insert(0, '.')
from comfyui_image_scorer.adapters.comfyui import NODE_CLASS_MAPPINGS
print('Nodes OK:', list(NODE_CLASS_MAPPINGS.keys()))
"                               # ComfyUI nodes load
# Start server briefly, verify no import errors
```

**Do not batch moves.** One file → validate → commit → next file.

---

## Phase 3: Move Domain Logic (High-Risk — One Subdomain at a Time)

Each subdomain move = **complete unit with its tests**. Follow same per-file validation.

### 3.1: `domain/comparison/` (from `external_modules/comparison/algorithm/`)
- Move all 9 files → `domain/comparison/`
- Fix imports to absolute
- **Critical:** Remove `database_structure` imports — replace with repository ports (interfaces) defined in `domain/database/ports/`
- Update `crystal_graph.py` to accept data via params (not import DB)

### 3.2: `domain/database/` (from `external_modules/database_structure/` non-endpoints)
- Move `schema.py`, `images_table.py`, `comparisons_table.py`, `path_handler.py`, `folder_organizer.py`, `deduplicate_scored.py`, `cleanup_orphans.py`
- Define **ports** (interfaces) in `domain/database/ports/`:
  - `ImageRepository`, `ComparisonRepository`, `PathResolver`
- Implement in `infrastructure/persistence/sqlite/`

### 3.3: `domain/loading/` (from `shared/loaders/`)
- Move `model_loader.py`, `maps_loader.py`, `training_loader.py`

### 3.4: `domain/graph/` (from `shared/graph/`)
- Move `crystal_graph.py`, `chain_manager.py`, `node_proxy.py`, `chain_proxy.py`, `component_proxy.py`
- **Fix:** Remove `external_modules` imports from `crystal_graph.py` — pass data via `rebuild_from_database(images, comparisons)`

### 3.5: `domain/vectors/` (from `shared/vectors/`)
- Move all vector files

### 3.6: `domain/analysis/` (from `shared/analysis/` + `external_modules/analysis/helpers.py`)
- Move `image_analysis.py`, `attribute_analysis.py`, `mediapipe_analysis.py`, `helpers.py`

### 3.7: `domain/training/` (from `shared/training/` + `external_modules/training_hyperparameters/` non-endpoints)
- Move `model_trainer.py`, `calibration.py`, `parameter_analysis.py`, `pair_data.py`, `plot.py`, `matrix_analysis.py`, `data_transformer.py`
- Move `config_utils.py`, `run.py`, `hyperparameter_optimizer.py`, `text_data/numerical_analysis.py`

### 3.8: `domain/data_transformation/` (from `external_modules/data_transform/` non-endpoints)
- Move `prepare_data.py`, `config/maps.py`, `data/processing.py`, `features/meta.py`

### Post-Each-Subdomain Validation (Must Pass)
```bash
pytest                           # All tests pass
pyright                          # No type errors
python -c "
import sys; sys.path.insert(0, '.')
from comfyui_image_scorer.adapters.comfyui import NODE_CLASS_MAPPINGS
print('Nodes OK:', list(NODE_CLASS_MAPPINGS.keys()))
"
# Start server, verify no import errors
# Run any domain-specific tests for moved module
```

---

## Phase 4: Move Adapters

### 4.1: `adapters/server/` (from `external_modules/server/` + `external_modules/*/endpoints.py`)
- `server/server.py` → `adapters/server/main.py`
- `server/image_processor.py` → `adapters/server/processor.py`
- Each `external_modules/*/endpoints.py` → `adapters/server/endpoints/<feature>.py`
- Create `adapters/server/routing/blueprints.py` for registration

### 4.2: `adapters/comfyui/` (from `nodes/`)
- `nodes/aesthetic_score/` → `adapters/comfyui/nodes/aesthetic_score/`
- Create `adapters/comfyui/__init__.py` with `NODE_CLASS_MAPPINGS` export
- Create `input_adapters/`, `output_adapters/`, `node_registry.py`
- **Rule:** Nodes import only from `application.services` and `core` — never `domain` directly

### 4.3: `adapters/cli/` (new)
- Create `main.py` with Typer/Click
- Create `commands/` from server/training/vectors/database logic

### Post-Phase 4 Validation (Must Pass)
```bash
pytest                           # All tests pass
pyright                          # No type errors
python -c "
import sys; sys.path.insert(0, '.')
from comfyui_image_scorer.adapters.comfyui import NODE_CLASS_MAPPINGS
print('Nodes OK:', list(NODE_CLASS_MAPPINGS.keys()))
for n, c in NODE_CLASS_MAPPINGS.items():
    assert hasattr(c, 'INPUT_TYPES') and hasattr(c, 'RETURN_TYPES')
print('All node contracts valid')
"
# Start server — verify all endpoints register
# Run CLI: comfyui-scorer --help
```

---

## Phase 5: Infrastructure Implementations

### 5.1: `infrastructure/persistence/sqlite/`
- Implement `ImageRepository`, `ComparisonRepository`, `PathResolver` from `domain/database/ports/`
- Use moved SQLite code from `domain/database/`

### 5.2: `infrastructure/ml_models/`
- Wrap ONNX/TFLite/Task models implementing `domain.loading` ports

### 5.3: `infrastructure/external_services/`
- HuggingFace, HTTP clients

### Post-Phase 5 Validation (Must Pass)
```bash
pytest                           # All tests pass
pyright                          # No type errors
# Start server — verify DB operations work
# Run CLI database commands
```

---

## Phase 6: Wire Dependency Injection

- Create `comfyui_image_scorer/__init__.py` with `create_container()` or similar
- Wire: `domain` ports → `infrastructure` implementations
- Pass implementations to `application.services` constructors
- Adapters receive `application.services` instances

### Post-Phase 6 Validation (Must Pass)
```bash
pytest                           # All tests pass
pyright                          # No type errors
python -c "
import sys; sys.path.insert(0, '.')
# Full import chain test
from comfyui_image_scorer.core import configuration, filesystem, observability, io, utilities
from comfyui_image_scorer.domain import comparison, database, graph, vectors, analysis, training, loading
from comfyui_image_scorer.application import services
from comfyui_image_scorer.adapters import comfyui, server
print('Full import chain OK')
"
# Start server — verify DI works
# ComfyUI node load test
```

---

## Phase 7: Configuration & Entry Points

### 7.1: `pyproject.toml`
```toml
[project]
name = "comfyui-image-scorer"
version = "0.1.0"
dependencies = [...]

[project.scripts]
comfyui-scorer = "comfyui_image_scorer.adapters.cli.main:app"
```

### 7.2: `config/` files — read via `core/configuration/settings.py`

### Post-Phase 7 Validation (Must Pass)
```bash
pytest                           # All tests pass
pyright                          # No type errors
pip install -e .                 # Editable install works
comfyui-scorer --help            # CLI entry point works
# Start server via CLI: comfyui-scorer server
```

---

## Phase 8: Final Validation Gates

| Check | Command |
|---|---|
| Type safety | `pyright` |
| Tests | `pytest` |
| Import structure | `pyright --verify-types comfyui_image_scorer` |
| Dependency sync | `uv pip compile pyproject.toml --output-file=- | diff - requirements.txt` |
| Architecture | `python -c "import comfyui_image_scorer; from comfyui_image_scorer.core import *; from comfyui_image_scorer.domain import *; ..."` |
| ComfyUI nodes | Programmatic verification from README |
| Server start | `comfyui-scorer server --port 8080` (background, then kill) |
| CLI commands | All subcommands execute without import errors |

**All must pass 100%.**

---

## Phase 9: Cleanup (Only After Phase 8 Passes)

- Delete `shared/`, `external_modules/`, `nodes/` (old locations)
- Delete `typings/` (no longer needed)
- Delete `maps2/` (duplicate)
- Verify `output/` and `downloaded_models/` work with new paths

### Post-Cleanup Validation
```bash
pytest
pyright
# Full end-to-end: ComfyUI loads nodes, server starts, CLI works
```

---

## Rollback Plan (At Any Phase)

If validation fails and cannot be fixed in 30 minutes:
1. `git stash` or `git checkout <phase-start-tag>` to pre-phase state
2. Tag each phase start: `git tag phase-2-start`
3. Re-apply phase in smaller increments (one file at a time)
4. **Do not proceed to next phase until current phase passes all validation gates**

---

## Notes

- **Colocated tests move with their modules** — keep `tests/` next to tested file
- **No relative imports anywhere** — absolute from `comfyui_image_scorer.*`
- **No `sys.path` manipulation** — editable install handles resolution
- **README.md is the contract** — if implementation deviates, fix implementation, not README
- **Validation commands are not optional** — they are the definition of "phase complete"