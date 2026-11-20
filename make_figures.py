"""
make_figures.py - makes the figures and tables for my report from outputs/results.csv.

  python make_figures.py

outputs/figures/:
  grid_<experiment>.png   rows = sketches, columns = input sketch + each setting
  curve_cond_scale.png    sketch recall and CLIP score vs ControlNet scale
  curve_guidance.png      same vs CFG guidance scale
  curve_img2img.png       same vs img2img strength
outputs/tables/:
  summary.csv             mean and std over sketches for every experiment and value
  <experiment>.tex        same numbers as LaTeX tables for Overleaf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from sketch2img import load_and_resize  # noqa: E402

# colours for the lines (one per sketch) and text
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 200,
})
LABELS = {"recall": "Sketch recall @5 px  (higher = structure kept)",
          "chamfer_px": "Chamfer distance [px]  (lower = structure kept)",
          "clip_score": "CLIP score  (higher = matches prompt)"}
XLABELS = {"cond_scale": "ControlNet conditioning scale", "guidance": "CFG guidance scale",
           "strength": "img2img strength", "style": "style", "control": "control type",
           "seed": "seed"}


def grid(df, exp, sketch_files, out):
    d = df[df.experiment == exp]
    sketches = list(dict.fromkeys(d.sketch))
    values = list(dict.fromkeys(d.value))
    n_r, n_c = len(sketches), len(values) + 1
    fig, axes = plt.subplots(n_r, n_c, figsize=(1.9 * n_c, 1.9 * n_r + 0.3), squeeze=False)
    for i, sk in enumerate(sketches):
        axes[i, 0].imshow(load_and_resize(sketch_files[sk]))
        axes[i, 0].set_title("input sketch" if i == 0 else "", fontsize=8, color=INK)
        for j, v in enumerate(values, start=1):
            row = d[(d.sketch == sk) & (d.value == v)].iloc[0]
            axes[i, j].imshow(Image.open(row.image))
            if i == 0:
                axes[i, j].set_title(f"{d.varied.iloc[0]} = {v}", fontsize=8, color=INK)
            axes[i, j].text(0.02, 0.02, f"R={row['recall']:.2f}", transform=axes[i, j].transAxes,
                            fontsize=6.5, color="white",
                            bbox=dict(facecolor="black", alpha=0.55, pad=1.5, linewidth=0))
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{out}/grid_{exp}.png", bbox_inches="tight")
    plt.close(fig)


def curve(df, exp, out):
    d = df[df.experiment == exp].copy()
    if d.empty:
        return
    d["value"] = d.value.astype(float)
    metrics = [m for m in ["recall", "clip_score"] if d[m].notna().any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 2.8), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for k, (sk, g) in enumerate(d.groupby("sketch", sort=False)):
            g = g.sort_values("value")
            ax.plot(g.value, g[m], color=SERIES[k % len(SERIES)], lw=1.2, marker="o", ms=4,
                    alpha=0.8, label=sk)
        mean = d.groupby("value")[m].mean()
        ax.plot(mean.index, mean.values, color=INK, lw=2, ls="--", marker="s", ms=4, label="mean")
        ax.set_xlabel(XLABELS.get(d.varied.iloc[0], d.varied.iloc[0]))
        ax.set_title(LABELS[m], fontsize=8.5, color=INK, loc="left")
    axes[0, -1].legend(frameon=False, fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out}/curve_{exp}.png", bbox_inches="tight")
    plt.close(fig)


def tables(df, out):
    cols = ["recall", "chamfer_px", "clip_score"]
    s = (df.groupby(["experiment", "varied", "value"], sort=False)[cols]
         .agg(["mean", "std"]).round(3))
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.reset_index().to_csv(f"{out}/summary.csv", index=False)
    for exp, g in df.groupby("experiment", sort=False):
        t = g.groupby("value", sort=False)[cols].agg(["mean", "std"])
        fmt = pd.DataFrame(index=t.index)
        for c, nd in zip(cols, [3, 2, 1]):
            fmt[c] = [f"{m:.{nd}f} $\\pm$ {sd:.{nd}f}" for m, sd in
                      zip(t[(c, "mean")], t[(c, "std")].fillna(0))]
        fmt.index.name = XLABELS.get(g.varied.iloc[0], g.varied.iloc[0])
        fmt.columns = ["Recall@5px $\\uparrow$", "Chamfer [px] $\\downarrow$", "CLIP $\\uparrow$"]
        with open(f"{out}/{exp}.tex", "w") as f:
            f.write(fmt.to_latex(escape=False, column_format="l" + "c" * len(cols)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    files = {os.path.splitext(os.path.basename(s["file"]))[0]: s["file"] for s in cfg["sketches"]}

    df = pd.read_csv(f"{a.out}/results.csv")
    fig_dir, tab_dir = f"{a.out}/figures", f"{a.out}/tables"
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(tab_dir, exist_ok=True)
    for exp in df.experiment.unique():
        grid(df, exp, files, fig_dir)
    for exp in ["cond_scale", "guidance", "img2img"]:
        curve(df, exp, fig_dir)
    tables(df, tab_dir)
    print(f"figures -> {fig_dir}/   tables -> {tab_dir}/")


if __name__ == "__main__":
    main()
