"""Run the whole redo analysis in order. Use the repo .venv (scikit-learn 1.7.2, the
version the isolation-forest model was saved with):

    .venv\\Scripts\\python.exe yucatan\\experiments\\redo\\run_all.py

Outputs: yucatan/outputs/redo/{results.md, results.json, figs/}; intermediate caches
(daily stacks as .npy, zone maps, per-step json) in yucatan/outputs/redo/cache (git-ignored).
Total runtime ~1 h on the 32-core workstation (stats ~7 min per method).
"""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STEPS = [["sigma.py"], ["iforest.py"], ["kmeans.py"], ["dbscan.py"],
         ["stats.py", "sigma", "iforest", "kmeans", "kmeans_month"], ["figures.py"], ["report.py"]]

for step in STEPS:
    print(f"\n=== {' '.join(step)}", flush=True)
    subprocess.run([sys.executable, str(HERE / step[0]), *step[1:]], cwd=HERE, check=True)
print("\nAll steps done.")
