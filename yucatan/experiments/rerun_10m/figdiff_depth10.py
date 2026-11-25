"""Share of pixels differing between outputs/rerun_10m/figs and outputs/redo/figs_depth10 -> json for summary_depth10.py."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Y = Path(__file__).resolve().parents[2]
A, B = Y / "outputs/rerun_10m/figs", Y / "outputs/redo/figs_depth10"
out = {}
for f in sorted(B.glob("*.png")):
    g = A / f.name
    if not g.exists():
        out[f.name] = "new"
        continue
    a, b = np.asarray(Image.open(g).convert("RGB")), np.asarray(Image.open(f).convert("RGB"))
    if a.shape != b.shape:
        out[f.name] = "size differs"
        continue
    d = (np.abs(a.astype(int) - b.astype(int)).sum(-1) > 30).mean()
    out[f.name] = "identical" if d == 0 else f"{100 * d:.2f}% px"
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=0))
