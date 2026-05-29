"""Execute Match_vs_Mismatch_Bias_Analysis.ipynb programmatically."""
import nbformat as nbf
from pathlib import Path
import os

NOTEBOOK_PATH = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\1_Code\Python_for_Check\Visualization\Match_vs_Mismatch_Bias_Analysis.ipynb")

nb = nbf.read(NOTEBOOK_PATH, as_version=4)
os.chdir(str(NOTEBOOK_PATH.parent))

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        print(f"\n[Cell {i}] {cell.cell_type} - skipped")
        continue
    source = cell.source
    first_line = source.strip().split('\n')[0][:100]
    print(f"\n[Cell {i}] {first_line}...")
    try:
        exec(source, globals())
    except Exception as e:
        print(f"  ERROR in cell {i}: {e}")
        import traceback
        traceback.print_exc()

print("\n\n=== ALL CELLS EXECUTED ===")
