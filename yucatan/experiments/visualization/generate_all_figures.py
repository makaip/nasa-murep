"""Run every figure generator using the existing precomputed Bayth pickle."""

from pathlib import Path
import subprocess
import sys


VISUALIZATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = VISUALIZATION_DIR.parents[2]
SCRIPTS = [
    "histograms/weekly.py",
    "raw/main.py",
    "scatterplots/monthly.py",
    "scatterplots/pooled.py",
    "scatterplots/yearly.py",
    "violins/all_years.py",
    "violins_all_years/main.py",
    "yearly_class/IF_bayth_all_years.py",
    "yearly_class/IF_bayth_across_years.py",
    "raw/bayth.py",
]


for relative_script in SCRIPTS:
    script = VISUALIZATION_DIR / relative_script
    print(f"\nRunning {relative_script}...")
    subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=True)

print("\nAll figures generated successfully.")
