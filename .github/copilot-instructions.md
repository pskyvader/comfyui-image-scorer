# Development Instructions for ComfyUI Trainer
# NEVER TOUCH THE VIRTUAL ENVIRONMENT FILES OR FOLDERS!
# this is NOT a GITHUB project. 
# These instructions are for GitHub Copilot use only.
## Always Do This
- All of these files show be used as reference always:
    - [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)
    - [config.json](../config.json)
    - requirements.txt
    - copilot-instructions.md (this file)
- do all task before asking anything.
- always follow the instructions in todo.md and done.md
- check and fix code style issues (see ## Error Handling)
- Run test sequence after making changes. (#testing process section below)
- See [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) for detailed file structure with all modules, classes, and functions.


## Common Commands
- always use the present virtual environment to any script execution, except for Jupyter notebooks
- Activate virtual environment: `.venv\Scripts\activate.ps1` (Windows PowerShell)
- any python code: `.venv\\Scripts\\python`
- when running code, always run in the format of `.venv\Scripts\python <path-to-script>` not as a module (no -m flag).

## Reuse Existing Modules
When adding new features, check [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) for available functions to reuse. 
Avoid duplicating functionality.
when adding new features, always update the structure file
when adding new features, check if existing modules can use the new functionality to avoid code duplication.
after adding new features, test using #testing process below.

## testing
- there should always be at least one test per method or function.
- there should be at least one file per module. the file should be in the same folder level as the module folder, named `test_<module_name>.py`.
- Ensure error handling covers edge cases (bad JSON, missing files, etc.)

# testing process
always follow ## Common Commands first to activate the virtual environment.
then follow these steps:
1. run the tests after any change (pytest)
2. when running tests, ensure all tests pass before continue
3. after all tests pass, Remove generated files after modifications
4. run the main code in each main directory to ensure no runtime errors in this precise order:
    - `ranking/`: `.venv/Scripts/python ranking/score_server.py --test-run` (verify config and exit)
    - `prepare/`: `.venv/Scripts/python prepare/prepare_data.py --limit 10` (use `--rebuild` to force regeneration)
    - `text_data/`: `.venv/Scripts/python text_data/prepare_text_data.py` (use `--rebuild` to force regeneration)
    - `training/`: run `training/training.ipynb`, `training/hyperparameter_optimize_loop.ipynb` and `training/feature_importance_analysis.ipynb` notebooks.
5. after everything runs without errors, update the `PROJECT_STRUCTURE.md` file if any new files, methods, or classes were added.
6. Check for missing libraries add any new one to `requirements.txt` if needed.



## Error Handling
- never use silent failures, or default fallbacks
- always fix any type of error, no matter how irrelevant they seem (for example: type hints warnings, unused variables, etc.)
- never hide or ignore errors (like tagging with `# noqa` or `# type: ignore`)
- **STRICT CONFIGURATION**: Never use `.get(key, default)` or `value = d[k] if k in d else default`. All configuration keys MUST exist. If a key is missing, let it raise `KeyError`. This enforces config schema validity. Do not hardcode "fallback" values in the code.
