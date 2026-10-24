#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stack the five per-case composite panels into one tall figure.

Reads ``$HRPCA_SHAP_OUT/FigS10_combo/FigS10_{a-e}_{id}_combo.png`` (written by
``combo_shap_mri.py``) and writes

* ``$HRPCA_SHAP_OUT/FigS10_all_cases.png`` — full size (4412 x 5462 px)
* ``$HRPCA_SHAP_OUT/FigS10_all_cases_2600px.png`` — 2600 px wide, for review

Paths
-----
``HRPCA_SHAP_OUT``   output root; reads ``FigS10_combo/`` and writes the two stacks

Usage
-----
    python src/shap/stack_5cases.py
"""
import os

from PIL import Image

P_OUT = os.environ.get("HRPCA_SHAP_OUT", "")


def need(value: str, var: str) -> str:
    if not value:
        raise SystemExit(f"{var} is not set — see src/shap/README.md")
    return value


ROOT = need(P_OUT, "HRPCA_SHAP_OUT")
SRC = os.path.join(ROOT, "FigS10_combo")
OUT = os.path.join(ROOT, "FigS10_all_cases.png")

CASES = ["a", "b", "c", "d", "e"]
GAP = 34

ims = []
for t in CASES:
    p = os.path.join(SRC, f"FigS10_{t}_combo.png")
    im = Image.open(p).convert("RGB")
    ims.append(im)
    print(f"({t}): {im.size}")

W = max(im.width for im in ims)
ims = [im.resize((W, int(im.height * W / im.width)), Image.LANCZOS) for im in ims]
H = sum(im.height for im in ims) + GAP * (len(ims) - 1)
canvas = Image.new("RGB", (W, H), "white")
y = 0
for im in ims:
    canvas.paste(im, (0, y))
    y += im.height + GAP
canvas.save(OUT, dpi=(300, 300))
print(f"\nstacked: {canvas.size} -> {OUT}")

# a 2600 px wide copy for review
W2 = 2600
c2 = canvas.resize((W2, int(canvas.height * W2 / canvas.width)), Image.LANCZOS)
p2 = OUT.replace(".png", "_2600px.png")
c2.save(p2, dpi=(300, 300))
print(f"scaled:  {c2.size} -> {p2}")
