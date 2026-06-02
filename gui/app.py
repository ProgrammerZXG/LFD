"""
LFD GUI - Interactive structural modeling tool
Usage: python gui/app.py  (run from the LFD/ sub-directory, i.e. cd LFD && python ../gui/app.py)
"""

import sys
import os
import io
import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import scipy.ndimage

# ── Resolve paths so we can import LFD modules ────────────────────────────────
GUI_DIR   = Path(__file__).resolve().parent        # .../LFD/gui/
LFD_DIR   = GUI_DIR.parent / "LFD"                # .../LFD/LFD/
sys.path.insert(0, str(LFD_DIR))

# ── Color helpers (mirrors engine.py) ─────────────────────────────────────────
def get_strata_colors(alpha=1):
    rgba = np.full((256, 4), 0, dtype=np.float32)
    strata = np.array([
        [1.0,0.0,0.0,alpha],[1.0,0.5019608,0.0,alpha],[1.0,1.0,0.0,alpha],
        [0.0,1,0.0,alpha],[0.0,0.5019608,0.0,alpha],[0.0,0.2509804,0.0,alpha],
        [0,1.0,1.0,alpha],[0.0,0.5019608,1.0,alpha],[0.0,0.0,1.0,alpha],
        [0.0,0.0,0.627451,alpha],[0.0,0.5019608,0.7529412,alpha],[1.0,0.5019608,0.5019608,alpha],
        [0.5019608,0.5019608,1.0,alpha],[0.5019608,0.0,1.0,alpha],[0.5019608,0,0.5019608,alpha],
        [1.0,0.5019608,1.0,alpha],[1.0,0.0,1.0,alpha],[0.5019608,0.2509804,0,alpha],
        [0.5019608,0.5019608,0.5019608,alpha],[0.7529412,0.7529412,0.7529412,alpha],
        [0.2509804,0,0.2509804,alpha],
        [0.90588236,0.7294118,0.19607843,alpha],[0.44313726,0.58431375,0.58431375,alpha],
        [0.5254902,0.42352942,0.4862745,alpha],[0.7176471,0.54509807,0.44313726,alpha],
        [0.5019608,0.5019608,0,alpha],[0.7529412,0.7294118,0.8784314,alpha],
        [0.61960787,0.85882354,0.9882353,alpha],[0.7372549,0.25882354,0.24705882,alpha],
        [0.8862745,0.8509804,0.627451,alpha],[0.60784316,0.9411765,0.7490196,alpha],
        [0.62352943,0.79607844,0.105882354,alpha],
    ], dtype=np.float32)
    c0 = np.array([0xFD/255.0, 0xC2/255.0, 0x3E/255.0], dtype=np.float32)
    c1 = np.array([0x6A/255.0, 0xAF/255.0, 0xE6/255.0], dtype=np.float32)
    c2 = np.array([0x67/255.0, 0xD5/255.0, 0xB5/255.0], dtype=np.float32)
    strata[2, :3] = c0; strata[18, :3] = c1; strata[19, :3] = c2
    for i in range(32):
        rgba[i*8:(i+1)*8] = strata[i]
    return ListedColormap(rgba)

# Horizon display colours (one per horizon index, cycling)
HORIZON_COLORS = [
    "#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
    "#00BCD4", "#8BC34A", "#FF5722", "#607D8B",
]
CANVAS_SIZE = 512   # logical size of the drawing canvas

# ══════════════════════════════════════════════════════════════════════════════
class DrawingCanvas(tk.Canvas):
    """Left-side canvas for drawing faults (lines) and horizons (curves)."""

    def __init__(self, master, size=CANVAS_SIZE, canvas_bg="#FFFFFF", **kwargs):
        super().__init__(master, width=size, height=size,
                         bg=canvas_bg, cursor="crosshair", **kwargs)
        self.size   = size
        self.mode   = None          # "fault" | "horizon" | None
        self.faults  = []           # list of (x0,y0,x1,y1)

        # horizons: list of completed horizons, each horizon = list of strokes,
        # each stroke = list of (x,y) points
        self.horizons = []          # [ [ [(x,y),...], [(x,y),...] ], ... ]
        self._current_strokes = []  # strokes of the horizon being built
        self._live_pts = []         # points of the stroke currently being drawn

        self._fault_first = None
        self._preview_line = None

        self.bind("<Button-1>",       self._on_click)
        self.bind("<B1-Motion>",      self._on_drag)
        self.bind("<ButtonRelease-1>",self._on_release)
        self.bind("<Motion>",         self._on_motion)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_mode(self, mode):
        self.mode = mode
        self._fault_first = None
        self._live_pts = []
        if self._preview_line:
            self.delete(self._preview_line)
            self._preview_line = None

    def end_horizon(self):
        """Commit the current in-progress horizon (may have multiple strokes)."""
        if self._current_strokes:
            self.horizons.append(list(self._current_strokes))
            self._current_strokes = []
        self._live_pts = []
        self.mode = None
        self._redraw()

    def has_active_horizon(self):
        return bool(self._current_strokes) or self.mode == "horizon"

    def smooth_current_horizon(self):
        """Smooth each stroke of the horizon currently being built."""
        smoothed = []
        for stroke in self._current_strokes:
            smoothed.append(_smooth_stroke(stroke))
        self._current_strokes = smoothed
        self._redraw()

    def smooth_last_committed(self):
        """Smooth each stroke of the last committed horizon."""
        if not self.horizons:
            return
        self.horizons[-1] = [_smooth_stroke(s) for s in self.horizons[-1]]
        self._redraw()

    def clear_all(self):
        self.faults.clear()
        self.horizons.clear()
        self._current_strokes = []
        self._live_pts = []
        self._fault_first = None
        self._preview_line = None
        self.delete("all")

    def to_numpy(self):
        """Convert drawn strokes → fx [H,W] and hrz [H,W] arrays."""
        fx  = np.zeros((self.size, self.size), dtype=np.float32)
        hrz = np.zeros((self.size, self.size), dtype=np.float32)

        for (x0, y0, x1, y1) in self.faults:
            _draw_line_on_array(fx, int(x0), int(y0), int(x1), int(y1), value=1.0, thickness=3)

        # Each horizon gets a unique integer label; strokes within same horizon share label
        for idx, strokes in enumerate(self.horizons, start=1):
            for stroke in strokes:
                for i in range(len(stroke) - 1):
                    x0, y0 = stroke[i]; x1, y1 = stroke[i+1]
                    _draw_line_on_array(hrz, int(x0), int(y0), int(x1), int(y1),
                                        value=float(idx), thickness=3)
        return fx, hrz

    # ── Event handlers ────────────────────────────────────────────────────────
    def _on_click(self, event):
        x, y = event.x, event.y
        if self.mode == "fault":
            if self._fault_first is None:
                self._fault_first = (x, y)
            else:
                x0, y0 = self._fault_first
                self.faults.append((x0, y0, x, y))
                self._fault_first = None
                if self._preview_line:
                    self.delete(self._preview_line)
                    self._preview_line = None
                self._redraw()
                self.mode = None
        elif self.mode == "horizon":
            self._live_pts = [(x, y)]

    def _on_drag(self, event):
        if self.mode == "horizon" and self._live_pts:
            self._live_pts.append((event.x, event.y))
            self._redraw_live()

    def _on_release(self, event):
        if self.mode == "horizon" and len(self._live_pts) >= 2:
            self._current_strokes.append(list(self._live_pts))
            self._live_pts = []
            self._redraw()
            # Stay in horizon mode so user can draw more strokes

    def _on_motion(self, event):
        if self.mode == "fault" and self._fault_first:
            if self._preview_line:
                self.delete(self._preview_line)
            x0, y0 = self._fault_first
            self._preview_line = self.create_line(
                x0, y0, event.x, event.y,
                fill="#ff4444", width=2, dash=(6, 3))

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def _horizon_color(self, idx):
        return HORIZON_COLORS[idx % len(HORIZON_COLORS)]

    def _redraw(self):
        self.delete("all")
        for (x0, y0, x1, y1) in self.faults:
            self.create_line(x0, y0, x1, y1, fill="#e74c3c", width=3)
        for idx, strokes in enumerate(self.horizons):
            color = self._horizon_color(idx)
            for stroke in strokes:
                if len(stroke) >= 2:
                    flat = [c for p in stroke for c in p]
                    self.create_line(*flat, fill=color, width=2, smooth=True)
        # In-progress horizon (committed strokes + live stroke)
        if self._current_strokes or self._live_pts:
            n = len(self.horizons)
            color = self._horizon_color(n)
            for stroke in self._current_strokes:
                if len(stroke) >= 2:
                    flat = [c for p in stroke for c in p]
                    self.create_line(*flat, fill=color, width=2, smooth=True)

    def _redraw_live(self):
        self.delete("live_stroke")
        if len(self._live_pts) >= 2:
            n = len(self.horizons)
            color = self._horizon_color(n)
            flat = [c for p in self._live_pts for c in p]
            self.create_line(*flat, fill=color, width=2,
                             smooth=True, tags="live_stroke")


# ══════════════════════════════════════════════════════════════════════════════
def _smooth_stroke(stroke, sigma=5):
    """Gaussian-smooth a single stroke (list of (x,y) tuples)."""
    if len(stroke) < 4:
        return stroke
    xs = scipy.ndimage.gaussian_filter1d([p[0] for p in stroke], sigma=sigma)
    ys = scipy.ndimage.gaussian_filter1d([p[1] for p in stroke], sigma=sigma)
    return list(zip(xs.tolist(), ys.tolist()))


def _draw_line_on_array(arr, x0, y0, x1, y1, value=1.0, thickness=3):
    """Bresenham line onto a 2-D numpy array (row=y, col=x)."""
    H, W = arr.shape
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    t = thickness // 2
    while True:
        for dr in range(-t, t+1):
            for dc in range(-t, t+1):
                r = y0 + dr; c = x0 + dc
                if 0 <= r < H and 0 <= c < W:
                    arr[r, c] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 <  dx: err += dx; y0 += sy


# ══════════════════════════════════════════════════════════════════════════════
class LFDApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("LFD – Structural Modeling GUI")
        self.configure(bg="#0f0f23")
        self.resizable(True, True)

        self._build_ui()
        self._generated_images = []
        self._raw_preds = []       # list of np arrays [H,W]
        self._cond_arrays = {}     # {"fx": np, "hrz": np}

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top toolbar ──────────────────────────────────────────────────────
        BG       = "#D4DFE6"   # main background
        TOOLBAR  = "#2C3E50"   # toolbar background
        BTN_BG   = "#B8C9D4"   # normal button
        BTN_HOV  = "#9DB4C0"   # button hover
        BTN_GEN  = "#2ECC71"   # generate button
        BTN_GENH = "#27AE60"
        FG_LIGHT = "#1A252F"   # dark text on light buttons
        FG_LABEL = "#2C3E50"   # dark text on light bg
        FG_SUB   = "#5D6D7E"   # secondary label
        ACCENT   = "#2980B9"   # accent / separator
        CANVAS_BG = "#FFFFFF"  # drawing canvas white

        self.configure(bg=BG)

        toolbar = tk.Frame(self, bg=TOOLBAR, pady=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_style = dict(bg=BTN_BG, fg=FG_LIGHT, relief=tk.FLAT,
                         padx=12, pady=4, font=("Helvetica", 11, "bold"),
                         activebackground=BTN_HOV, activeforeground=FG_LIGHT,
                         cursor="hand2")

        self.btn_fault = tk.Button(toolbar, text="＋ Add Fault",
                                   command=self._start_fault, **btn_style)
        self.btn_fault.pack(side=tk.LEFT, padx=6)

        self.btn_horizon = tk.Button(toolbar, text="＋ Add Horizon",
                                     command=self._start_horizon, **btn_style)
        self.btn_horizon.pack(side=tk.LEFT, padx=6)

        self.btn_end_horizon = tk.Button(toolbar, text="✔ End Horizon",
                                         command=self._end_horizon,
                                         bg="#5D6D7E", fg=FG_LIGHT, relief=tk.FLAT,
                                         padx=12, pady=4, font=("Helvetica", 11, "bold"),
                                         activebackground="#4A5568", activeforeground=FG_LIGHT,
                                         cursor="hand2", state=tk.DISABLED)
        self.btn_end_horizon.pack(side=tk.LEFT, padx=6)

        self.btn_smooth = tk.Button(toolbar, text="⌁ Smooth",
                                    command=self._smooth_horizon, **btn_style)
        self.btn_smooth.pack(side=tk.LEFT, padx=6)

        self.btn_clear = tk.Button(toolbar, text="✕ Clear",
                                   command=self._clear, **btn_style)
        self.btn_clear.pack(side=tk.LEFT, padx=6)

        # Separator
        tk.Frame(toolbar, bg=ACCENT, width=2).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Label(toolbar, text="Samples:", bg=TOOLBAR, fg="#ECF0F1",
                 font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.n_samples_var = tk.IntVar(value=1)
        spin = ttk.Spinbox(toolbar, from_=1, to=50, width=4,
                           textvariable=self.n_samples_var, font=("Helvetica", 11))
        spin.pack(side=tk.LEFT, padx=4)

        tk.Label(toolbar, text="CFG:", bg=TOOLBAR, fg="#ECF0F1",
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(8,0))
        self.cfg_var = tk.DoubleVar(value=1.0)
        cfg_spin = ttk.Spinbox(toolbar, from_=0.5, to=10.0, increment=0.5,
                                width=5, textvariable=self.cfg_var,
                                font=("Helvetica", 11))
        cfg_spin.pack(side=tk.LEFT, padx=4)

        tk.Label(toolbar, text="Steps:", bg=TOOLBAR, fg="#ECF0F1",
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(8,0))
        self.steps_var = tk.IntVar(value=20)
        steps_spin = ttk.Spinbox(toolbar, from_=5, to=100, increment=5,
                                  width=4, textvariable=self.steps_var,
                                  font=("Helvetica", 11))
        steps_spin.pack(side=tk.LEFT, padx=4)

        self.btn_save = tk.Button(toolbar, text="💾 Save Results",
                                  command=self._save_results,
                                  bg=BTN_BG, fg=FG_LIGHT, relief=tk.FLAT,
                                  padx=12, pady=4, font=("Helvetica", 11, "bold"),
                                  activebackground=BTN_HOV, activeforeground=FG_LIGHT,
                                  cursor="hand2", state=tk.DISABLED)
        self.btn_save.pack(side=tk.RIGHT, padx=6)

        self.btn_gen = tk.Button(toolbar, text="▶  Generate",
                                 command=self._run_generation,
                                 bg=BTN_GEN, fg=FG_LIGHT, relief=tk.FLAT,
                                 padx=16, pady=4, font=("Helvetica", 11, "bold"),
                                 activebackground=BTN_GENH, activeforeground=FG_LIGHT,
                                 cursor="hand2")
        self.btn_gen.pack(side=tk.RIGHT, padx=10)

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready. Draw faults and horizons, then click Generate.")
        status_bar = tk.Label(self, textvariable=self.status_var,
                              bg=TOOLBAR, fg="#ECF0F1",
                              font=("Helvetica", 10), anchor=tk.W, padx=8)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Main area ────────────────────────────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Left panel: drawing canvas
        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        tk.Label(left, text="Conditions", bg=BG, fg=FG_LABEL,
                 font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0,4))

        self.canvas = DrawingCanvas(left, size=CANVAS_SIZE,
                                    canvas_bg=CANVAS_BG,
                                    highlightthickness=2,
                                    highlightbackground=ACCENT)
        self.canvas.pack()

        # Mode indicator
        self.mode_label = tk.Label(left, text="Mode: —",
                                   bg=BG, fg=FG_SUB,
                                   font=("Helvetica", 10))
        self.mode_label.pack(anchor=tk.W, pady=(4,0))

        # Legend
        legend = tk.Frame(left, bg=BG)
        legend.pack(anchor=tk.W, pady=4)
        tk.Canvas(legend, width=18, height=4, bg="#e74c3c",
                  highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(legend, text=" Fault", bg=BG, fg=FG_LABEL,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0,10))
        tk.Canvas(legend, width=18, height=4, bg="#27ae60",
                  highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(legend, text=" Horizon", bg=BG, fg=FG_LABEL,
                 font=("Helvetica", 10)).pack(side=tk.LEFT)

        # Right panel: results
        right = tk.Frame(main, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        result_header = tk.Frame(right, bg=BG)
        result_header.pack(fill=tk.X, pady=(0,4))
        tk.Label(result_header, text="Generated Models", bg=BG, fg=FG_LABEL,
                 font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self.btn_clear_results = tk.Button(result_header, text="✕ Clear Results",
                                           command=self._clear_results,
                                           bg=BTN_BG, fg=FG_LIGHT, relief=tk.FLAT,
                                           padx=8, pady=2, font=("Helvetica", 10),
                                           activebackground=BTN_HOV, activeforeground=FG_LIGHT,
                                           cursor="hand2")
        self.btn_clear_results.pack(side=tk.RIGHT)

        # Scrollable frame for results
        self.result_frame_outer = tk.Frame(right, bg=BG)
        self.result_frame_outer.pack(fill=tk.BOTH, expand=True)

        self.result_canvas = tk.Canvas(self.result_frame_outer,
                                       bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.result_frame_outer, orient=tk.VERTICAL,
                                  command=self.result_canvas.yview)
        self.result_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.result_inner = tk.Frame(self.result_canvas, bg=BG)
        self.result_canvas_window = self.result_canvas.create_window(
            (0, 0), window=self.result_inner, anchor=tk.NW)

        self.result_inner.bind("<Configure>", self._on_result_configure)
        self.result_canvas.bind("<Configure>", self._on_canvas_configure)

        # Progress bar (hidden until generation starts)
        self.progress = ttk.Progressbar(right, mode="indeterminate", length=300)
        self._BG = BG

    # ── Button callbacks ──────────────────────────────────────────────────────
    def _start_fault(self):
        self.canvas.set_mode("fault")
        self.mode_label.config(text="Mode: Fault — click two points")
        self.status_var.set("Fault mode: click point 1, then point 2.")

    def _start_horizon(self):
        self.canvas.set_mode("horizon")
        self.btn_end_horizon.config(state=tk.NORMAL)
        self.mode_label.config(text="Mode: Horizon — drag to draw, End Horizon when done")
        self.status_var.set("Horizon mode: drag to draw strokes; click End Horizon to finish this horizon.")

    def _end_horizon(self):
        self.canvas.end_horizon()
        self.btn_end_horizon.config(state=tk.DISABLED)
        self.mode_label.config(text="Mode: —")
        self.status_var.set(f"Horizon committed. Total horizons: {len(self.canvas.horizons)}.")

    def _smooth_horizon(self):
        if self.canvas.has_active_horizon():
            self.canvas.smooth_current_horizon()
            self.status_var.set("Current horizon strokes smoothed.")
        else:
            self.canvas.smooth_last_committed()
            self.status_var.set("Last horizon smoothed.")

    def _clear(self):
        """Clear everything: canvas + results."""
        self.canvas.clear_all()
        self.canvas.set_mode(None)
        self.btn_end_horizon.config(state=tk.DISABLED)
        self.mode_label.config(text="Mode: —")
        self._clear_results()
        self.status_var.set("Cleared.")

    def _clear_results(self):
        for w in self.result_inner.winfo_children():
            w.destroy()
        self._generated_images.clear()
        self._raw_preds.clear()
        self._cond_arrays.clear()
        self.btn_save.config(state=tk.DISABLED)
        self.status_var.set("Results cleared.")

    def _save_results(self):
        if not self._raw_preds:
            return
        from tkinter import filedialog
        out_dir = filedialog.askdirectory(title="Select folder to save results")
        if not out_dir:
            return
        out_path = Path(out_dir)

        strata_cmap = get_strata_colors()
        hrz_norm = self._cond_arrays.get("hrz")
        black_cmap = ListedColormap(["black"])
        black_cmap.set_bad(alpha=0)

        for i, pred in enumerate(self._raw_preds):
            # Save npy
            np.save(str(out_path / f"sample_{i:03d}.npy"), pred)
            # Save png
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            ax.imshow(pred, cmap=strata_cmap, vmin=-1, vmax=1)
            if hrz_norm is not None:
                ax.imshow(np.ma.masked_where(hrz_norm <= -0.999, hrz_norm),
                          cmap=black_cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(str(out_path / f"sample_{i:03d}.png"), dpi=100,
                        bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        # Save condition masks
        if "fx" in self._cond_arrays:
            np.save(str(out_path / "cond_fx.npy"), self._cond_arrays["fx"])
        if "hrz" in self._cond_arrays:
            np.save(str(out_path / "cond_hrz.npy"), self._cond_arrays["hrz"])

        self.status_var.set(f"Saved {len(self._raw_preds)} samples + conditions to {out_dir}")

    # ── Generation ────────────────────────────────────────────────────────────
    def _run_generation(self):
        if not self.canvas.faults and not self.canvas.horizons:
            messagebox.showwarning("No input",
                "Please draw at least one fault or horizon before generating.")
            return

        # Check model exists
        ckpt_dir = LFD_DIR / "model" / "LFD_test"
        ckpt_file = ckpt_dir / "checkpoint-last.pth"
        if not ckpt_file.exists():
            messagebox.showerror("Model not found",
                f"Checkpoint not found:\n{ckpt_file}\n\n"
                "Download the model and place it at model/LFD_test/checkpoint-last.pth")
            return

        self.btn_gen.config(state=tk.DISABLED)
        self.progress.pack(fill=tk.X, padx=4, pady=4)
        self.progress.start(12)
        self.status_var.set("Generating… please wait.")

        # Clear old results
        for w in self.result_inner.winfo_children():
            w.destroy()
        self._generated_images.clear()

        # Run in background thread
        thread = threading.Thread(target=self._generate_thread, daemon=True)
        thread.start()

    def _generate_thread(self):
        try:
            import torch
            from denoiser import DenoiserFH as Denoiser
            import scipy.ndimage as ndi

            n_samples = self.n_samples_var.get()
            cfg       = float(self.cfg_var.get())

            # ── Build condition arrays ────────────────────────────────────────
            fx_raw, hrz_raw = self.canvas.to_numpy()   # [512,512] each

            # Normalise fx: non-zero → 1, else → -1
            fx = np.full_like(fx_raw, -1.0)
            fx[fx_raw != 0] = 1.0

            # Normalise hrz: use depth-weighted mapping (same as inference.py)
            hrz_norm = _normalize_hrz(hrz_raw)

            # Stack → [1, 2, 512, 512]
            cond = np.stack([fx, hrz_norm], axis=0)[None].astype(np.float32)

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            self.after(0, lambda: self.status_var.set(
                f"Using device: {device}. Loading model…"))

            # ── Load model ────────────────────────────────────────────────────
            import argparse
            args = argparse.Namespace(
                model="LFD-B/32", img_size=512,
                attn_dropout=0.0, proj_dropout=0.0,
                class_num=1, in_channels=1, cond_in_ch=2,
                P_mean=-1.0, P_std=0.8, noise_scale=0.1, t_eps=5e-2,
                label_drop_prob=0.0, ema_decay1=0.9999, ema_decay2=0.9996,
                sampling_method="heun", num_sampling_steps=self.steps_var.get(),
                cfg=cfg, interval_min=0.1, interval_max=1.0,
                pretrained_base="",
            )
            model = Denoiser(args).to(device)

            ckpt_path = str(LFD_DIR / "model" / "LFD_test" / "checkpoint-last.pth")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_ema1" in ckpt:
                sd = ckpt["model_ema1"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                sd = ckpt["model"]
            else:
                sd = ckpt
            model.load_state_dict(sd, strict=False)
            model.eval()

            cond_t = torch.from_numpy(cond).to(device)

            # ── Generate in batches ───────────────────────────────────────────
            batch_size = min(10, n_samples)
            n_batches  = math.ceil(n_samples / batch_size)
            all_preds  = []

            fx_thick  = ndi.maximum_filter(fx,      size=2)
            hrz_thick = ndi.maximum_filter(hrz_norm, size=2)

            with torch.no_grad():
                for b in range(n_batches):
                    bs = min(batch_size, n_samples - b * batch_size)
                    self.after(0, lambda b=b, nb=n_batches:
                        self.status_var.set(f"Generating batch {b+1}/{nb}…"))
                    labels = torch.zeros(bs, dtype=torch.long, device=device)
                    bc     = cond_t.repeat(bs, 1, 1, 1)
                    preds  = model.generate(labels, bc)   # [bs,1,512,512]
                    preds_np = preds.cpu().numpy()

                    # smooth + fault-fill (same post-proc as inference.py)
                    preds_np = _smooth(preds_np)
                    preds_np = _replace_fault(preds_np, fx_thick)
                    all_preds.extend([preds_np[i, 0] for i in range(bs)])

            # ── Render to memory only (no auto-save) ─────────────────────────
            strata_cmap = get_strata_colors()
            black_cmap = ListedColormap(["black"])
            black_cmap.set_bad(alpha=0)

            tk_images = []
            for pred in all_preds:
                fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
                ax.imshow(pred, cmap=strata_cmap, vmin=-1, vmax=1)
                ax.imshow(np.ma.masked_where(hrz_norm <= -0.999, hrz_norm),
                          cmap=black_cmap, vmin=-1, vmax=1, interpolation="nearest")
                ax.axis("off")
                fig.tight_layout(pad=0)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100,
                            bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf).resize((256, 256), Image.LANCZOS).copy()
                tk_images.append(ImageTk.PhotoImage(img))

            cond_data = {"fx": fx, "hrz": hrz_norm}
            self.after(0, lambda: self._show_results(tk_images, all_preds, cond_data))

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            msg = str(exc)
            self.after(0, lambda m=msg, t=tb: self._generation_error(m, t))

    def _show_results(self, tk_images, raw_preds, cond_data):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_gen.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.status_var.set(f"Done! {len(tk_images)} samples generated. Click Save Results to export.")

        self._generated_images = tk_images
        self._raw_preds = list(raw_preds)
        self._cond_arrays = cond_data

        cols = 4
        for i, img in enumerate(tk_images):
            r, c = divmod(i, cols)
            frm = tk.Frame(self.result_inner, bg=self._BG,
                           highlightthickness=1, highlightbackground="#A0B4C0")
            frm.grid(row=r, column=c, padx=4, pady=4)
            lbl = tk.Label(frm, image=img, bg=self._BG)
            lbl.pack()
            tk.Label(frm, text=f"#{i:03d}", bg=self._BG, fg="#5D6D7E",
                     font=("Helvetica", 9)).pack()

    def _generation_error(self, msg, tb):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_gen.config(state=tk.NORMAL)
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Generation failed", f"{msg}\n\n{tb}")

    # ── Canvas resize helpers ─────────────────────────────────────────────────
    def _on_result_configure(self, event):
        self.result_canvas.configure(
            scrollregion=self.result_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.result_canvas.itemconfig(
            self.result_canvas_window, width=event.width)


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing helpers (mirrors inference.py, no imports needed from there)

def _normalize_hrz(hrz_raw: np.ndarray) -> np.ndarray:
    """Depth-weighted normalisation of horizon label array → [-1, 1]."""
    labels = np.unique(hrz_raw)
    labels = labels[labels != 0]
    if labels.size == 0:
        return np.full_like(hrz_raw, -1.0, dtype=np.float32)
    H = hrz_raw.shape[0]
    depth_idx = np.indices(hrz_raw.shape, dtype=np.int32)[0].astype(np.float32)
    mapping = {}
    for lab in labels:
        z_mean = depth_idx[hrz_raw == lab].mean()
        v = np.clip(-1.0 + 2.0 * z_mean / max(H - 1, 1), -1.0, 1.0)
        mapping[lab] = np.float32(v)
    out = np.full_like(hrz_raw, -1.0, dtype=np.float32)
    for lab, val in mapping.items():
        out[hrz_raw == lab] = val
    return out


def _smooth(pred_np, sigma=1.0):
    import scipy.ndimage as ndi
    out = pred_np.copy()
    for b in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[b, c] = ndi.gaussian_filter(out[b, c], sigma=sigma, mode="nearest")
    return out


def _replace_fault(pred_np, fault_np, thr=-0.999, dilate=1):
    import scipy.ndimage as ndi
    mask = fault_np > thr
    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=dilate)
    if not np.any(mask):
        return pred_np
    _, (iy, ix) = ndi.distance_transform_edt(mask.astype(np.uint8),
                                              return_indices=True)
    out = pred_np.copy()
    for b in range(out.shape[0]):
        for c in range(out.shape[1]):
            img = out[b, c]
            img[mask] = img[iy[mask], ix[mask]]
            out[b, c] = img
    return out


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = LFDApp()
    app.mainloop()
