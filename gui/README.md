# LFD GUI

Interactive tool for drawing faults/horizons and generating structural models.

## Requirements

Same conda environment as the main project (`jit`), plus:

```bash
pip install pillow --break-system-packages
```

## Usage

Run from the **repository root** (`LFD/` folder containing `LFD/`, `gui/`, etc.):

```bash
cd /path/to/LFD          # the outer repo root
conda activate jit
python gui/app.py
```

> The script automatically adds `LFD/` to `sys.path` so it can import the model code.

## Workflow

| Step | Action |
|------|--------|
| 1 | Click **＋ Add Fault** then click two points on the canvas to draw a red fault line. Repeat as needed. |
| 2 | Click **＋ Add Horizon** then drag the mouse to draw a horizon curve. |
| 3 | Optionally click **⌁ Smooth Last Horizon** to smooth the most recently drawn horizon. |
| 4 | Set the number of **Samples** and **CFG** scale in the toolbar. |
| 5 | Click **▶ Generate** — results appear in the right panel and are saved to `LFD/output/gui_run/`. |

CUDA is used automatically if available; falls back to CPU otherwise.
