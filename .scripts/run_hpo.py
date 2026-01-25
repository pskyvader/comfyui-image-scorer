from AI_jobs.run_notebooks import run_notebook
from pathlib import Path
import traceback

try:
    res = run_notebook(Path('full_data/training/hyperparameter_optimize_loop.ipynb'), timeout=600)
    print('RESULT_OK')
    print(res)
except Exception as e:
    print('ERROR')
    traceback.print_exc()
