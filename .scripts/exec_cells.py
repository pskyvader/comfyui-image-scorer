import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from pathlib import Path

nb_path = Path('full_data/training/hyperparameter_optimize_loop.ipynb')
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name="python3")

print(f"Executing notebook: {nb_path}")
client.setup_kernel()

for i, cell in enumerate(nb.cells, start=1):
    print(f"\n--- Cell {i} ({cell.get('cell_type')}) ---")
    try:
        if cell.get('cell_type') == 'code':
            out = client.execute_cell(cell, i-1)
            # Print outputs
            for o in cell.get('outputs', []):
                if 'text' in o:
                    print(o['text'])
        else:
            print(cell.get('source'))
    except CellExecutionError as e:
        print(f"Cell {i} raised CellExecutionError:")
        print(e)
        break
    except Exception as e:
        print(f"Cell {i} raised Exception:")
        import traceback
        traceback.print_exc()
        break

print('Execution finished (stopped on error or completed)')
