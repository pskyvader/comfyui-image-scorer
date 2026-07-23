# ComfyUI Image Scorer — Architecture Documentation

## Overview

This custom node provides aesthetic scoring, pairwise comparison ranking, gallery browsing, latent map visualization, and hyperparameter optimization for generated images. The codebase follows a **strict layered architecture** with **dependency inversion**: dependencies point inward only.

```
core → domain → application → adapters → infrastructure
```

No layer may import from a layer to its right. The **ComfyUI node integration is the primary deliverable**; all other layers exist to serve it.

---

## Project Root Layout

| Path | Purpose |
|---|---|
| `comfyui_image_scorer/` | Python package (installable source) |
| `config/` | Runtime configuration files (JSON) — read-only at startup |
| `downloaded_models/` | Downloaded third-party model weights (MediaPipe `.task`, `.tflite`) — input only |
| `output/` | **Regeneratable runtime state** (git-ignored). Safe to `rm -rf`. Contains: SQLite DB, vector caches, generated maps, exported models. |
| `pyproject.toml` / `uv.lock` / `requirements.txt` | Build and dependency metadata |
| `pyrightconfig.json` | Static type checker configuration |

---

## Package Structure (`comfyui_image_scorer/`)

### `core/` — Kernel
**Zero internal dependencies.** Pure utilities used by every layer.
- `configuration/` — Settings loading, validation, defaults
- `filesystem/` — Path registry, resolution, runtime directories
- `observability/` — Structured logging, correlation IDs
- `io/` — Serialization, atomic writes, binary helpers
- `utilities/` — Pure functions (collections, math, text, time)

**Rule:** `core` imports **nothing** from `domain`, `application`, `adapters`, or `infrastructure`.

---

### `domain/` — Domain Layer
**Depends only on `core`.** Contains all business logic, algorithms, data structures, and repository interfaces. No framework code (no Flask, no ComfyUI, no SQLAlchemy).
- `comparison/` — TrueSkill rating, pairwise state, phase ordering, graph helpers
- `database/` — Repository interfaces, entities, schema definitions
- `data_transformation/` — Feature pipelines, metadata normalization, map configs
- `training/` — HPO orchestration, calibration, parameter analysis
- `analysis/` — Image/attribute analysis, MediaPipe integration
- `graph/` — Crystal graph, chain management, proxy objects
- `vectors/` — Embedding, keypoint, position, person-map vectors
- `loading/` — Model loaders (aesthetic, MediaPipe, maps)

**Rule:** `domain` defines **ports** (interfaces) for persistence, external APIs, and ML runtimes. Implementations live in `infrastructure/`.

---

### `application/` — Application Layer
**Depends on `core` + `domain`.** Orchestrates domain objects into use cases. Thin, stateless services.
- `services/` — RankingService, GalleryService, MapService, TrainingService, AnalysisService
- `dto/` — Data transfer objects crossing adapter boundaries
- `ports/` — Abstract interfaces for adapters to implement (optional, for clarity)

**Rule:** No Flask, no ComfyUI, no SQL. Pure orchestration.

---

### `adapters/` — Adapter Layer (Framework Boundaries)
**Depends on `core` + `domain` + `application`.** Translates framework protocols → domain calls.

#### `adapters/server/` — Flask REST API
- `routing/` — Blueprint registration, URL prefixes
- `endpoints/` — Thin request/response handlers (validation → service call → JSON)
- `middleware/` — Error handling, CORS, request logging

#### `adapters/comfyui/` — ComfyUI Node Integration (Primary Deliverable)
- `__init__.py` — Exports `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`
- `nodes/` — Node implementations grouped by feature
  - `aesthetic_score/` — Scoring nodes
  - `ranking/` — Comparison/rating nodes
  - `gallery/` — Browser nodes
  - `maps/` — Latent map visualization nodes
- `input_adapters/` — ComfyUI types (IMAGE, LATENT, STRING) → Domain DTOs
- `output_adapters/` — Domain results → ComfyUI types
- `node_registry.py` — Central registration, category management

**Rule:** Nodes contain **zero domain logic** — only translation and delegation to `application.services`.

#### `adapters/cli/` — Command-Line Interface
**Depends on `core` + `domain` + `application`.** Translates shell commands → service calls.
- `main.py` — Entry point (Typer/Click/argparse), subcommand router
- `commands/`
  - `server.py` — Start Flask server
  - `training.py` — Run training / HPO
  - `vectors.py` — Generate vectors
  - `database.py` — DB maintenance (cleanup, dedup, vacuum)
- `output.py` — Rich/typer formatting helpers

**Exposed via `pyproject.toml`:**
```toml
[project.scripts]
comfyui-scorer = "comfyui_image_scorer.adapters.cli.main:app"
```

**Usage after `pip install -e .`:**
```bash
comfyui-scorer server --port 8080
comfyui-scorer training run --config config/training_config.json
comfyui-scorer vectors generate --input output/images --output output/vectors
comfyui-scorer database cleanup --vacuum
```

---

## `infrastructure/` — Infrastructure Implementations
**Depends on `core` + `domain` (implements domain ports).** Concrete adapters for external systems.
- `persistence/` — SQLite repositories implementing `domain.database` ports
- `ml_models/` — ONNX/TFLite/Task wrappers implementing `domain.loading` ports
- `external_services/` — HuggingFace, HTTP clients, etc.

---

## Dependency Rules (Enforced by Architecture)

| Layer | May Import From | Must Not Import From |
|---|---|---|
| `core` | (stdlib only) | `domain`, `application`, `adapters`, `infrastructure` |
| `domain` | `core` | `application`, `adapters`, `infrastructure` |
| `application` | `core`, `domain` | `adapters`, `infrastructure` |
| `adapters/*` | `core`, `domain`, `application` | other `adapters/*`, `infrastructure` |
| `infrastructure` | `core`, `domain` | `application`, `adapters` |

**Violations are architectural errors.** Use dependency injection (pass implementations as arguments) to cross boundaries outward.

---

## Runtime Data Rules

| Directory | Content | Lifecycle | Git |
|---|---|---|---|
| `config/` | User-editable JSON settings | Persistent, user-managed | Tracked |
| `downloaded_models/` | Downloaded third-party weights (MediaPipe `.task`, `.tflite`) | Persistent, downloaded once | Ignored (large) |
| `output/` | SQLite DB, vector caches, generated maps, exported models | **Ephemeral** — safe to delete anytime | **Ignored** |

**Golden rule:** If `rm -rf output/` requires zero manual steps to recover, it belongs in `output/`.

---

## ComfyUI Integration Contract

- Entry point: `comfyui_image_scorer.adapters.comfyui.__init__`
- Exports: `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`
- Nodes declare `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY` per ComfyUI spec
- All node logic delegates to `application.services.*`
- Type translation only in `input_adapters/` and `output_adapters/`

---

## Development Conventions

1. **Imports:** Absolute from package root (`from comfyui_image_scorer.core.configuration import Settings`). No `sys.path` manipulation. No relative imports (`from ....core...`). **Why:** Relative imports break when running files as scripts, when entry points differ (ComfyUI vs. server), or when restructuring. Absolute imports work everywhere once the package is installed in editable mode (`pip install -e .`). Both ComfyUI nodes and the server entry point resolve correctly.
2. **Tests:** Colocated `tests/` subdirectory next to tested module (e.g., `domain/comparison/tests/test_trueskill.py`).
3. **Typing:** Full type hints on public APIs. `pyright --level=basic` must pass.
4. **No global mutable state** in `core`/`domain`/`application`. State lives in `adapters` or `infrastructure`.
5. **Configuration** enters only via `core.configuration` — no `os.getenv` scattered in domain code.

---

## Node Import Verification

ComfyUI discovers nodes at startup. Verify registration works:

**1. Startup logs** — ComfyUI prints:
```
[ComfyUI] Loaded custom node: comfyui_image_scorer
[ComfyUI] Registered nodes: AestheticScorerLoader, CalculateAestheticScore, ...
```

**2. Node menu** — Right-click canvas → search node name → appears under its `CATEGORY`.

**3. Programmatic test** (run in CI / locally):
```bash
# From project root
python -c "
import sys
sys.path.insert(0, '.')
from comfyui_image_scorer.adapters.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
print('Nodes:', list(NODE_CLASS_MAPPINGS.keys()))
for name, cls in NODE_CLASS_MAPPINGS.items():
    assert hasattr(cls, 'INPUT_TYPES'), f'{name}: missing INPUT_TYPES'
    assert hasattr(cls, 'RETURN_TYPES'), f'{name}: missing RETURN_TYPES'
    assert hasattr(cls, 'FUNCTION'), f'{name}: missing FUNCTION'
    assert hasattr(cls, 'CATEGORY'), f'{name}: missing CATEGORY'
    print(f'  ✓ {name}: {cls.CATEGORY} / {cls.FUNCTION}')
print('All nodes valid.')
```

**4. Unit test** (colocated):
```python
# adapters/comfyui/nodes/aesthetic_score/tests/test_node_import.py
from comfyui_image_scorer.adapters.comfyui import NODE_CLASS_MAPPINGS

def test_aesthetic_nodes_registered():
    expected = {"AestheticScorerLoader", "CalculateAestheticScore"}
    actual = set(NODE_CLASS_MAPPINGS.keys())
    assert expected.issubset(actual), f"Missing: {expected - actual}"
```

---

## Development Rules (from AGENTS.md)

### Engineering Style
- **Small, direct changes** — touch the narrowest code path that explains the bug/behavior
- **Minimize files changed** — a change touching many files is suspect
- **Practical over architectural** — add abstractions only when they remove real duplication or match existing ComfyUI patterns
- **Fewer dependencies** — no new deps unless absolutely necessary
- **Delete dead code aggressively** — no fallbacks, migration paths, debug prints, or compatibility branches that aren't needed
- **Revert broken behavior fast** — better to remove a broken feature than keep a partial fix
- **Preserve APIs** — node names, model-loading behavior, file layout, workflow compatibility unless explicitly replacing them
- **No AI-generated code style** — no unnecessary helper layers, vague names, boilerplate comments, defensive branches without real failure modes, broad rewrites

### Architecture Boundaries
- **Layer focus** — each layer owns its concepts; don't leak UI, API, workflow, queue, persistence, telemetry, model-loading, node, or execution concerns across layers
- **Shared modules depend down** — only on lower primitives and own domain concepts
- **Narrow data across boundaries** — no broad context objects, request metadata, IDs, bookkeeping state, or callbacks unless the receiver genuinely needs them
- **Identify smallest owner layer** before touching many files

### No Internet Requests
- **No outbound network calls** from core/domain/application layers — no telemetry, analytics, tracking, usage reporting, crash reporting, update checks, remote config, feature flags, metrics, licensing checks
- **Model downloading only when explicitly user-initiated** — limited to requested artifact, no background activity
- **Local-only behavior allowed** — if it stays on user's machine with no network access, tracking, or persistent identification

### State Ownership
- **State lives on the object that owns the behavior** using it
- **No probing children** with `getattr(child, "...", default)` to decide parent control flow — if parent needs to branch on a capability, initialize an explicit parent-owned field when child is constructed/attached
- **Prefer direct attributes with clear defaults** over implicit feature detection through arbitrary child attributes
- **Child capability checks only when child owns the behavior** and parent is simply delegating

### Interface Contracts
- **Public methods = stable contracts** — don't change return shapes, add sentinel wrappers, or alter signatures without updating all callers
- **Preserve caller invocation** — required args, order, return type, side effects, error behavior
- **No compatibility params/flags** unless read by current code and changing behavior
- **No model-specific options in shared helpers** — keep one-off behavior at the integration boundary
- **Normalize third-party returns at the boundary** — core code sees expected types, not model-specific variants

### Autograd / Model Freezing
- **No `torch.no_grad`, `torch.inference_mode` wrappers** — ComfyUI models are always frozen for inference
- **No freeze/unfreeze/trainability toggles** on model classes
- **Remove training-only behavior** (dropout) from inference code; preserve checkpoint compatibility with `nn.Identity` if needed

### Python Style
- **Imports at module scope** — no inline imports unless for optional-backend probes or import cycles
- **No unnecessary try/except** — only for optional deps/platform/backends with a useful fallback
- **No version workarounds** for pinned library versions
- **Fail clearly** on unsupported formats, invalid quantization, bad state — no silent quality degradation
- **Match local file style** — long lines, simple helpers, module-level state, direct tensor ops are fine when clearer

### Model / Device / Memory
- **dtype, device, VRAM, offloading = core correctness** — check CPU/CUDA/ROCm/MPS/DirectML/XPU/NPU/low-VRAM implications
- **Use ComfyUI helpers** — `comfy.quant_ops`, `model_management`, `memory_management`, `pinned_memory`, `comfy-kitchen`
- **Use optimized kernels** — prefer shared ops over handwritten; adapt inputs to documented layout
- **All models use ComfyUI-selected attention** — treat backend as opaque; don't inspect function identity/names/modules
- **No custom ops duplicating existing ones with float32 upcast** — use generic ComfyUI ops / native torch
- **`operations` param in `__init__` is never `None`** — no fallback branches
- **No unnecessary params** in model/block/ops classes — only values actually used for inference
- **Reuse existing model classes/blocks/ops/helpers** before implementing new ones
- **Model detection uses first dimension only** — second dim may be half for NVFP4/4-bit
- **Guard every state-dict key** in detection — no partial match then KeyError
- **No `einops` in core inference** — use native torch `reshape`, `view`, `permute`, `transpose`, `flatten`, `unflatten`, `unsqueeze`, `squeeze`
- **No tensors for Python-side metadata** — sequence lengths, offsets, indices, counts stay as Python ints/lists
- **No unnecessary casts/transfers** — preserve compute dtype, storage dtype, bias dtype, shape metadata
- **Trust optimized backend dtype contract** — don't cast results back unless documented
- **Model-native latent layout stays in model** — no collapsing/expanding in nodes
- **DiT models: pad to patch size, crop output only** — use `comfy.ldm.common_dit.pad_to_patch_size`
- **No defensive shape checks** that just replace the tensor op's own error — validate only at real boundaries
- **Inputs to model forward = compute dtype** (except integer timesteps) — no convenience casts
- **Raw params not owned by ops: cast at use** with `comfy.ops.cast_to_input` or `model_management.cast_to`
- **Model code doesn't manage memory** — loading, offloading, device movement, VRAM policy, cache lifetime belong in execution/model-management layers
- **No global/module/class/singleton stores for tensors** across executions — temporary caches scoped to single forward/encode/decode call

### Nodes & User-Facing Behavior
- **Follow conventions**: `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, local registration mapping
- **Backward compatible by default** — add inputs with sensible defaults; don't change output types unless required
- **Minimal nodes** — reuse existing nodes; adapt model to existing nodes > create new nodes
- **`io.Autogrow`** for variable repeated inputs — min 0 when valid no-item path, cap only when real limit
- **Mark inputs optional** when execution has valid path without them
- **Conditioning nodes output conditioning only** — no convenience image outputs
- **Nodes output only what they own** — no pass-through outputs unless explicitly an output node
- **Nodes expose only inputs they read** — no placeholder/pass-through/compatibility inputs
- **Node code never patches model code directly** — use model patcher class
- **Warnings/info = short and actionable** — remove noisy messages rather than adding more

### Commit & Review
- **Subjects**: `Fix ...`, `Add ...`, `Support ...`, `Remove ...`, `Update ...`, `Make ...`, `Use ...`, `Disable ...`, `Bump ...`, `Revert ...`
- **PR descriptions**: problem, behavioral change, tests run — no long narratives
- **One coherent behavioral change per commit**
- **Review priority**: crashes, wrong dtype/device, memory regressions, broken model loading, workflow incompatibility, noisy/misleading output

---

## Continuous Integration

- **Dependency sync check:** `.github/workflows/check-deps.yml` — fails if `requirements.txt` drifts from `pyproject.toml`.
- Run locally: `uv pip compile pyproject.toml -o requirements.txt`

---

## Quick Start (Developer)

> **All commands below MUST run inside the ComfyUI virtual environment** (the one ComfyUI uses). Activate it first:
> ```bash
> # Windows (PowerShell)
> & "C:\path\to\ComfyUI\venv\Scripts\Activate.ps1"
> 
> # Linux/macOS
> source /path/to/ComfyUI/venv/bin/activate
> ```
> This ensures `torch`, `comfy`, and all ComfyUI-internal packages resolve correctly for both the server and node entry points.

```bash
# Install in editable mode
pip install -e .

# Run type checks
pyright

# Run tests
pytest

# ComfyUI: symlink this folder into ComfyUI/custom_nodes/
```

---

## License

MIT — see `LICENSE`.