#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stack a SHAP force plot (top) with its MRI habitat panel row (bottom).

The MRI row is supplied by the reader as an SVG whose panels are base64-embedded
PNGs (DWI / ADC / k = 2 … k = 9, with ``k = 5 (src)`` being the modelled one).
This script decodes those images, rescales the row to the width of the SHAP
figure, moves the panel labels underneath the panels and pastes the two rows
together.

**The MRI panels in Fig. 6 require the source images, which are not
distributed** — without them this script can only be run by someone who has the
imaging data.

Expected inputs
---------------
``$HRPCA_SHAP_CASES/FigS10_{tag}_{id}.png``
    the per-case SHAP plots, tags ``a``–``e``
``$HRPCA_SHAP_MRI/{id}{h}.svg``
    the MRI panel row for that case, ``h`` being the habitat count of the case
    (e.g. ``case_a_h4.svg``)

Set ``PANEL_ORDER=mri_top`` to put the MRI row on top; the default is SHAP on top.

Paths
-----
================================  ==========================================
``HRPCA_SHAP_OUT``                output root; writes ``FigS10_combo/``
``HRPCA_SHAP_CASES``              optional, defaults to ``$HRPCA_SHAP_OUT/cases``
``HRPCA_SHAP_MRI``                directory holding the ``{id}{h}.svg`` files
================================  ==========================================

Usage
-----
    python src/shap/combo_shap_mri.py
"""
import base64
import io
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- paths: set these in the environment (see src/shap/README.md) -----------
P_OUT = os.environ.get("HRPCA_SHAP_OUT", "")
P_CASES = os.environ.get("HRPCA_SHAP_CASES", "")
P_MRI = os.environ.get("HRPCA_SHAP_MRI", "")


def need(value: str, var: str) -> str:
    if not value:
        raise SystemExit(f"{var} is not set — see src/shap/README.md")
    return value


ROOT = need(P_OUT, "HRPCA_SHAP_OUT")
SHAP_DIR = P_CASES or os.path.join(ROOT, "cases")
MRI_DIR = need(P_MRI, "HRPCA_SHAP_MRI")
OUT = os.path.join(ROOT, "FigS10_combo")
os.makedirs(OUT, exist_ok=True)

# The MRI row for each case; the habitat count is part of the file name.
SVG = {
    "a": ("case_a", "case_a_h4.svg"),
    "b": ("case_b", "case_b_h4.svg"),
    "c": ("case_c", "case_c_h2.svg"),
    "d": ("case_d", "case_d_h2.svg"),
    "e": ("case_e", "case_e_h4.svg"),
}
TARGET_W = 4412          # match the width of the SHAP figure


def _font(size, bold=False):
    """Arial if matplotlib can find it, else Pillow's built-in bitmap font."""
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont(
            font_manager.FontProperties(family="Arial",
                                        weight="bold" if bold else "normal")), size)
    except Exception:
        return ImageFont.load_default()


def autocrop(im, tol=6, pad=6):
    """Trim the white margin (keeping ``pad`` pixels)."""
    a = np.array(im.convert("RGB")).astype(int)
    mask = (np.abs(a - 255).sum(axis=2) > tol)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return im
    r0 = max(0, rows[0] - pad)
    r1 = min(im.height, rows[-1] + 1 + pad)
    c0 = max(0, cols[0] - pad)
    c1 = min(im.width, cols[-1] + 1 + pad)
    return im.crop((c0, r0, c1, r1))


def render_mri(svg_path):
    """Parse the SVG, rebuild one MRI panel row; labels go underneath the panels."""
    txt = io.open(svg_path, encoding="utf-8").read()
    vb = re.search(r'viewBox="([\d\.\s]+)"', txt)
    vx, vy, vw, vh = [float(t) for t in vb.group(1).split()]
    s = TARGET_W / vw                                    # scale factor
    img_pat = (r'<image x="([\d\.]+)" y="([\d\.]+)" width="([\d\.]+)" height="([\d\.]+)"'
               r'[^>]*xlink:href="data:image/png;base64,([^"]+)"')
    match = list(re.finditer(img_pat, txt))
    img_bottom = max(float(m.group(2)) + float(m.group(4)) for m in match)
    LABEL_Y = img_bottom + 42                            # label baseline, below the panels
    canvas = Image.new("RGB", (int(vw * s), int((LABEL_Y + 16) * s)), "white")
    for m in match:
        x, y, w, h = (float(m.group(i)) for i in range(1, 5))
        im = Image.open(io.BytesIO(base64.b64decode(m.group(5)))).convert("RGB")
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        canvas.paste(im, (int((x - vx) * s), int((y - vy) * s)))
    # text labels -> below the panels (x still centred on each panel)
    dr = ImageDraw.Draw(canvas)
    for m in re.finditer(r'<text x="([\d\.]+)" y="([\d\.]+)"[^>]*font-size="([\d\.]+)"[^>]*>([^<]+)</text>', txt):
        x, fs, label = float(m.group(1)), float(m.group(3)), m.group(4)
        fnt = _font(max(12, int(fs * s * 1.8)), bold="bold" in m.group(0))
        bb = dr.textbbox((0, 0), label, font=fnt)
        dr.text(((x - vx) * s - (bb[2] - bb[0]) / 2, (LABEL_Y - vy) * s - (bb[3] - bb[1])),
                label, fill=(0, 0, 0), font=fnt)
    return canvas


made = []
for tag, (pid, svg_name) in SVG.items():
    shap_png = os.path.join(SHAP_DIR, f"FigS10_{tag}_{pid}.png")
    svg = os.path.join(MRI_DIR, svg_name)
    if not (os.path.exists(shap_png) and os.path.exists(svg)):
        print("missing file:", tag, pid)
        continue
    top = Image.open(shap_png).convert("RGB")
    top = autocrop(top, pad=4)
    bot = autocrop(render_mri(svg), pad=4)
    if top.width != TARGET_W:
        top = top.resize((TARGET_W, int(top.height * TARGET_W / top.width)), Image.LANCZOS)
    if bot.width != TARGET_W:
        bot = bot.resize((TARGET_W, int(bot.height * TARGET_W / bot.width)), Image.LANCZOS)
    gap = 6
    ORDER = os.environ.get("PANEL_ORDER", "shap_top")   # shap_top | mri_top
    W = TARGET_W
    H = top.height + gap + bot.height
    canvas = Image.new("RGB", (W, H), "white")
    if ORDER == "mri_top":
        canvas.paste(bot, (0, 0))
        canvas.paste(top, (0, bot.height + gap))
        suffix = "_MRItop"
    else:
        canvas.paste(top, (0, 0))
        canvas.paste(bot, (0, top.height + gap))
        suffix = ""
    fp = os.path.join(OUT, f"FigS10_{tag}_{pid}_combo{suffix}.png")
    canvas.save(fp, dpi=(300, 300))
    made.append((tag, pid, canvas.size, fp))
    print(f"  ({tag}) {pid}: SHAP {top.size} + MRI {bot.size} -> {canvas.size}")

print("\noutput:", OUT)
