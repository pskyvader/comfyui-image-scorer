from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

p = Path.cwd()
root = p
while not (root / "pyproject.toml").exists() and root.parent != root:
    root = root.parent
full_data_path = root / "full_data"
# Add root first so 'shared' package (at repo root) can be imported
sys.path.insert(0, str(root))
if full_data_path.exists():
    sys.path.insert(0, str(full_data_path))
print('sys.path[0]=', sys.path[0])
try:
    import importlib
    importlib.invalidate_caches()
    import training.data_utils
    import training.run
    print('Imported training.data_utils and training.run successfully')
except Exception as e:
    import traceback
    print('Failed to import training:', e)
    traceback.print_exc()
