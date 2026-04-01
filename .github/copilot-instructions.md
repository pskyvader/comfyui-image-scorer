# Development Instructions for ComfyUI Image Scorer

## ⚠️ CRITICAL CONSTRAINTS
- **ONLY modify files inside `custom_nodes/comfyui-image-scorer/` folder**
- **NEVER modify anything outside this folder** (even for reading Python files, you can read but not modify)
- **ALWAYS use ComfyUI's virtual environment** located at `e:/ComfyUI/.venv` (or user's ComfyUI root/.venv)
- **ALWAYS confirm you're inside the venv before running any code** - check prompt starts with `(.venv)`
- **Forbidden to modify**: ComfyUI core files, other custom nodes, system files

## Project Overview
**ComfyUI Image Scorer** is a comprehensive image aesthetic scoring system integrated as a ComfyUI custom node.
- **Location**: `e:\ComfyUI\custom_nodes\comfyui-image-scorer\`
- **Python Version**: 3.14+ (see pyproject.toml)
- **Dependencies**: Managed via `uv` tool (see pyproject.toml)

## Entry Points
The project has three main processing steps:
1. **Step 01 - Ranking** (`external_modules/step01ranking/`): Web UI for scoring images (Flask server)
2. **Step 02 - Prepare** (`external_modules/step02prepare/`): Data preparation (images + text)
3. **Step 03 - Training** (`external_modules/step03training/`): Model training & analysis (Jupyter notebooks)

## Always Reference These Files
- [todo.md](../todo.md) - Current task list (MUST follow these tasks)
- [config/config.json](../config/config.json) - Main configuration
- [config/prepare_config.json](../config/prepare_config.json) - Data preparation settings
- [config/training_config.json](../config/training_config.json) - Training parameters
- [config/vector_config.json](../config/vector_config.json) - Vector configuration
- [pyproject.toml](../pyproject.toml) - Dependencies and project metadata
- This file

## Virtual Environment Setup
**ABSOLUTE REQUIREMENT**: Always use ComfyUI's venv
- Workspace location: `e:\ComfyUI`
- Virtual environment: `e:\ComfyUI\.venv`
- Activate: `e:\ComfyUI\.venv\Scripts\activate.ps1` (Windows PowerShell)
- After activation, your prompt should show: `(.venv) PS> `

## Dependency Management
⚠️ Use `uv` tool, NOT pip directly:
```powershell
# Add new dependency
uv add <package-name>

# Remove dependency  
uv remove <package-name>

# Update all dependencies
uv update
```

After modifying `pyproject.toml`:
```powershell
pip install -e .
```

## Common Commands
**Always execute from project root with venv activated!**

```powershell
# Verify venv is active (should show: (.venv) PS>)
python --version

# Run Python scripts
python external_modules/step01ranking/score_server.py
python external_modules/step02prepare/full_data/prepare_data.py --rebuild --limit 10

# Run tests
pytest

# Run notebooks (requires Jupyter)
jupyter notebook external_modules/step03training/full_data/training.ipynb
```

## Code Style & Type Hints
- Use Python 3.9+ type hints:
  - ✅ `dict` instead of `Dict`
  - ✅ `list` instead of `List`
  - ✅ `tuple` instead of `Tuple`
  - ✅ `X | None` instead of `Optional[X]`
  - ✅ Keep `TypedDict`, `Literal`, `Any` from typing module
- All files must pass type checking with Pylance
- Never use `# type: ignore` or `# noqa` without explicit justification
- All errors must be fixed, not silenced

## Reuse Existing Modules
When adding new features:
1. Check [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) for available functions
2. Avoid duplicating functionality
3. Always update PROJECT_STRUCTURE.md when adding new functions/classes
4. Run test suite after changes


# test creation guidelines
- there should always be at least one test per method or function.
- there should be at least one file per module. the file should be in the same folder level as the module folder, inside a test folder, named `test_<module_name>.py`.
- Ensure error handling covers edge cases (bad JSON, missing files, etc.)

# testing process
**CRITICAL**: Always activate venv first before running tests!

```powershell
# Step 1: Confirm venv is active
(.venv) PS> python --version

# Step 2: Run all tests
(.venv) PS> pytest

# Step 3: When tests pass, run main pipeline validation
(.venv) PS> python external_modules/step01ranking/score_server.py --test-run

(.venv) PS> python external_modules/step02prepare/full_data/prepare_data.py --rebuild --limit 10

(.venv) PS> python external_modules/step02prepare/full_data/prepare_data.py

(.venv) PS> python external_modules/step02prepare/text_data/prepare_text_data.py --rebuild

# Step 4: Notebooks (if modified)
(.venv) PS> jupyter notebook external_modules/step03training/full_data/training.ipynb
```

**After each pipeline run**:
1. Verify no errors in output
2. Check that output files were created (vectors.jsonl, scores.jsonl, etc.)
3. Remove generated test outputs
4. Run full test suite to ensure no regressions

## Error Handling
- **STRICT CONFIGURATION**: Never use `.get(key, default)` or `value = d[k] if k in d else default`. All configuration keys MUST exist. If a key is missing, let it raise `KeyError`. This enforces config schema validity.
- Never use silent failures or default fallbacks
- Always fix any type of error, including:
  - Type hints warnings
  - Unused variables
  - Unused imports
  - Incorrect exception handling
  - Missing return statements
- Never hide or ignore errors (like tagging with `# noqa` or `# type: ignore` without explicit justification)
- All errors must be fixed, not worked around

## File Organization
```
custom_nodes/comfyui-image-scorer/
├── external_modules/
│   ├── step01ranking/        # Scoring UI server
│   ├── step02prepare/         # Data preparation
│   └── step03training/        # Model training
├── shared/                    # Shared utilities
├── nodes/                     # ComfyUI node implementations
├── tests/                     # Test suite
├── config/                    # Configuration files
├── output/                    # Generated outputs
└── ...
```

## When Making Changes
1. **Before starting**: Read the task description in todo.md
2. **During coding**: Follow all constraints above
3. **After coding**:
   - Run `pytest` to verify tests pass
   - Run pipeline validation steps
   - Update PROJECT_STRUCTURE.md if needed
   - Check for type hint warnings
   - Remove temporary/generated files
4. **When done**: Mark task as complete in todo.md
