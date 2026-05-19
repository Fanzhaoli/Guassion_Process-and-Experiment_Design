"""Execute the updated notebook via nbclient with all cached-results cells."""
import json
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_PATH = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_for_Check\Sigmoid_Optimized\v9_Sigmoid_Optimization.ipynb")

print("Loading notebook...")
nb = nbformat.read(str(NB_PATH), as_version=4)
print(f"Loaded {len(nb.cells)} cells")

# Execute
client = NotebookClient(nb, timeout=300, kernel_name='python3')

print("Executing notebook (using cached optimization results)...")
t0 = time.time()
try:
    import asyncio
    asyncio.run(client.async_execute())
    print(f"\nExecution complete in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"\nError during execution: {e}")
    import traceback
    traceback.print_exc()

# Save
nbformat.write(nb, str(NB_PATH))
print(f"Notebook saved to: {NB_PATH}")

# Stats
n_with_output = sum(1 for c in nb.cells if c.cell_type == 'code' and c.get('outputs'))
n_total_code = sum(1 for c in nb.cells if c.cell_type == 'code')
n_error = sum(1 for c in nb.cells if c.cell_type == 'code' and any(
    o.get('output_type') == 'error' for o in (c.get('outputs') or [])
))
print(f"Code cells: {n_total_code}, With outputs: {n_with_output}, Errors: {n_error}")