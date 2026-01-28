# Development Instructions for ComfyUI Trainer
# NEVER TOUCH THE VIRTUAL ENVIRONMENT FILES OR FOLDERS!
### AI-jobs: unless explicitly instructed, the only place to create new files for any extra task is in the folder ./AI_jobs
## Always Do This
- All of these files show be used as reference always:
    - [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)
    - [config/config.json](../config/config.json)
    - [config/prepare_config.json](../config/prepare_config.json)
    - [config/training_config.json](../config/training_config.json)
    - [todo.md](../todo.md)
    - pyproject.toml
    - this file
- when one task is done, move it from todo.md to done.md, and then immediately proceed to the next task. Dont wait for feedback or reviews until all tasks are done.
- always follow the instructions in todo.md and done.md
- check and fix code style issues (see ## Error Handling)
- Run test sequence after making changes. (#testing process section below)
- See [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) for detailed file structure with all modules, classes, and functions.

## Dependency Management
- always use `pyproject.toml` to add new dependencies.
- use uv tool to add new dependencies:
    - to add a new dependency: `uv add <package-name>` (e.g., `uv add requests`)
    - to remove a dependency: `uv remove <package-name>` (e.g., `uv remove requests`)
    - to update all dependencies: `uv update`
- after modifying `pyproject.toml`, run `pip install -e .` to update the virtual environment.

## Common Commands
- always use the present virtual environment to any script execution, except for Jupyter notebooks
- Activate virtual environment: `.venv\Scripts\activate.ps1` (Windows PowerShell)
- any python code: `python` (or `.venv\Scripts\python` only if not activated)
- when running code, always run in the format of `python <path-to-script>` not as a module (no -m flag).

## Reuse Existing Modules
When adding new features, check [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) for available functions to reuse. 
Avoid duplicating functionality.
when adding new features, always update the structure file
when adding new features, check if existing modules can use the new functionality to avoid code duplication.
after adding new features, test using #testing process below.

# test creation guidelines
- there should always be at least one test per method or function.
- there should be at least one file per module. the file should be in the same folder level as the module folder, inside a test folder, named `test_<module_name>.py`.
- Ensure error handling covers edge cases (bad JSON, missing files, etc.)

# testing process
always follow [## Common Commands] first to activate the virtual environment.
then follow these steps:
1. run the tests after any change (pytest)
2. when running tests, ensure all tests pass before continue
3. after all tests pass, Remove generated files after modifications
4. run the main code in each main directory to ensure no runtime errors in this precise order:
    - reset prepare config file (config/prepare_config.json), set vector schema/slots values to 1, (this is needed for testing all functions, including resize of the vectors).
    - `python step01ranking/score_server.py --test-run` (verify config and exit)
    - `python step02prepare/full_data/prepare_data.py --rebuild --limit 10` (first run with small limit to verify no runtime errors)
    - `python step02prepare/full_data/prepare_data.py` (second run full without limit nor rebuild to check full data prepare)
    - `python step02prepare/text_data/prepare_text_data.py --rebuild (use `--rebuild` to force regeneration)
    - `step03training/full_data/training.ipynb` 
    - `step03training/full_data/hyperparameter_optimize_loop.ipynb`     
    - `step03training/text_data/training.ipynb`
    - `python step04export/deploy.py` (to verify deployment process, and then confirm that the files exist in the folder)

5. after everything runs without errors, update the `PROJECT_STRUCTURE.md` file if any new files, methods, or classes were added.
6. Check for missing libraries add any new one to `pyproject.toml` if needed.



## Error Handling
- never use silent failures, or default fallbacks
- always fix any type of error, no matter how irrelevant they seem (for example: type hints warnings, unused variables, etc.)
- never hide or ignore errors (like tagging with `# noqa` or `# type: ignore`)
- **STRICT CONFIGURATION**: Never use `.get(key, default)` or `value = d[k] if k in d else default`. All configuration keys MUST exist. If a key is missing, let it raise `KeyError`. This enforces config schema validity. Do not hardcode "fallback" values in the code.
