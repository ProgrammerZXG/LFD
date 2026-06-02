# LFD GUI

Interactive tool for drawing faults/horizons and generating structural models.

## Requirements

Same conda environment as the main project (`jit`), plus:

```bash
pip install pillow --break-system-packages
```

## Usage

Run from the **repository root** (the folder containing `LFD/`, `gui/`, etc.):

```bash
cd /path/to/LFD          # outer repo root
conda activate jit
python gui/app.py
```

> The script automatically adds the `LFD/` subdirectory to `sys.path` so it can import model code.

## Workflow

| Step | Action |
|------|--------|
| 1 | Click **＋ Add Fault** → click two points on the canvas to draw a fault line. Repeat as needed. |
| 2 | Click **＋ Add Horizon** → drag to draw one or more strokes (strokes can be discontinuous). |
| 3 | Click **✔ End Horizon** to commit the current horizon. Repeat steps 2–3 for more horizons. |
| 4 | Optionally click **⌁ Smooth** to smooth the active or last committed horizon. |
| 5 | Set **Samples**, **CFG**, and **Steps** in the toolbar. Fewer steps = faster but lower quality. |
| 6 | Click **▶ Generate** — results appear in the right panel. |
| 7 | Click **💾 Save Results** to export PNG / NPY files and condition masks to a folder of your choice. |
| 8 | Use **✕ Clear Results** (next to "Generated Models") to clear only the results, or **✕ Clear** in the toolbar to reset everything (canvas + results). |

## Parameter Guide

| Parameter | Default | Recommendation |
|-----------|---------|----------------|
| **CFG** | 1.0 | Higher values (2.0–4.0) strengthen condition control, producing results that follow faults and horizons more strictly. |
| **Samples** | 1 | Set to higher values (e.g. 5–20) to generate multiple diverse realizations at once. |
| **Steps** | 20 | Set to 50 for best quality. Lower values are faster but may reduce output fidelity. |

> ⚠️ Increasing CFG, Samples, or Steps all raise GPU/CPU computation cost. On CPU, generating 1 sample at 50 steps may take several minutes.

## Notes

- CUDA is used automatically if available; falls back to CPU otherwise.
- MPS (Apple Silicon) is not supported due to operator compatibility issues.
- Results are **not** saved automatically — click **Save Results** to export.
- Saved files per run: `sample_000.png`, `sample_000.npy`, …, `cond_fx.npy`, `cond_hrz.npy`.
