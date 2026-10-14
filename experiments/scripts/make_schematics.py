"""Vector schematic of the CL-TransCPD framework (replaces the hand-drawn overview).

  python scripts/make_schematics.py [--out ../figures/generated]

The drawing is generated programmatically from the method definition; it contains no data and no
generated imagery. Plain journal style: serif type matching the manuscript, white boxes with thin outlines
in light-grey phase panels, one frame around the figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

INK = "#1A1A1A"
LINE = "#333333"
PANEL = "#F4F4F4"           # pretraining panel (neutral)
TINT = {"cond": "#E9EFF5",  # signal conditioning: soft blue-grey
        "enc": "#F6EFE4",   # Cross-Lead Transformer and change score: pale sand
        "diag": "#E9F1E8"}  # change-triggered diagnosis: light sage
NOTE = "#444444"


def _style():
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})


def box(ax, x, y, w, h, text, fs=7.0, weight="normal", fill="white", lw=0.8):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=LINE, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight, zorder=4,
            linespacing=1.25)


def panel(ax, x, y, w, h, title, fill=PANEL):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=LINE, lw=0.8, zorder=1))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="center", fontsize=8.0, weight="bold", color=INK, zorder=4)


def arrow(ax, x0, y0, x1, y1, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=LINE, lw=lw, ls=ls,
                                 zorder=2, shrinkA=0, shrinkB=0))


def elbow(ax, pts, lw=0.8):
    """Poly-line through pts with an arrow head on the last segment (routed through the gaps between panels)."""
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=LINE, lw=lw, zorder=2, solid_capstyle="butt")
    arrow(ax, *pts[-2], *pts[-1], lw=lw)


def down(ax, x, y_from, y_to):
    arrow(ax, x, y_from, x, y_to)


def framework(out: Path):
    _style()
    fig = plt.figure(figsize=(7.2, 4.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.6, 101.6)
    ax.set_ylim(-1.6, 65.5)
    ax.axis("off")
    ax.add_patch(Rectangle((-0.9, -0.9), 101.8, 65.7, fc="white", ec=INK, lw=1.0, zorder=0))   # one frame

    top, height = 17.5, 45.7

    # ---------------- signal conditioning
    panel(ax, 1.0, top, 24.0, height, "Signal conditioning phase", TINT["cond"])
    bx, bw = 3.0, 20.0
    box(ax, bx, 53.3, bw, 4.6, "12-lead ECG stream")
    box(ax, bx, 45.8, bw, 4.6, "Band-pass filter\n(0.5–40 Hz, 100 Hz)")
    box(ax, bx, 36.6, bw, 6.4, "IHRAS\nheart-rate-adaptive\ntwo-cycle segments")
    box(ax, bx, 28.6, bw, 5.2, "HTFP\nlearned time-frequency gate")
    box(ax, bx, 20.6, bw, 5.2, "RSSA\nsubspace denoising")
    for y0, y1 in ((53.3, 50.4), (45.8, 43.0), (36.6, 33.8), (28.6, 25.8)):
        down(ax, bx + bw / 2, y0, y1)

    # ---------------- Cross-Lead Transformer and change score
    panel(ax, 28.0, top, 39.0, height, "Cross-Lead Transformer and change-score phase", TINT["enc"])
    px, pw = 30.0, 35.0
    box(ax, px, 53.3, pw, 4.6, "Patch embedding\n12 leads × 16 patches, one CLS token per lead")
    ax.add_patch(Rectangle((px, 38.6), pw, 11.8, fc="white", ec=LINE, lw=0.8, zorder=3))
    ax.text(px + pw / 2, 48.6, "Dual-axis encoder (4 layers)", ha="center", va="center", fontsize=7.2,
            weight="bold", color=INK, zorder=4)
    sw, sg = 10.0, 1.6
    x0 = px + (pw - 3 * sw - 2 * sg) / 2
    for i, t in enumerate(("Temporal\nattention", "Cross-lead\nattention", "Feed-forward\nnetwork")):
        box(ax, x0 + i * (sw + sg), 39.8, sw, 6.6, t, fs=6.8)
    for i in (1, 2):
        arrow(ax, x0 + i * (sw + sg) - sg + 0.25, 43.1, x0 + i * (sw + sg) - 0.25, 43.1)
    lw_, gap = 16.5, pw - 2 * 16.5
    box(ax, px, 30.2, lw_, 5.6, "Segment representation\n" r"$\mathbf{h}_k$")
    box(ax, px + lw_ + gap, 30.2, lw_, 5.6, "Lead-wise decoder\n(reconstruction)")
    box(ax, px, 19.8, pw, 7.6, "Online change score\n"
        r"deviation $\Vert\mathbf{h}_k-\bar{\mathbf{h}}\Vert_2$ from the" "\n"
        r"moving-average reference $\bar{\mathbf{h}}$; alarm when $s_k>\gamma$")
    down(ax, px + pw / 2, 53.3, 50.4)
    down(ax, px + lw_ / 2, 38.6, 35.8)
    down(ax, px + lw_ + gap + lw_ / 2, 38.6, 35.8)
    down(ax, px + lw_ / 2, 30.2, 27.4)
    elbow(ax, [(bx + bw, 23.2), (26.5, 23.2), (26.5, 55.6), (px, 55.6)])

    # ---------------- change-triggered diagnosis
    panel(ax, 70.0, top, 29.0, height, "Change-triggered diagnosis phase", TINT["diag"])
    qx, qw = 72.0, 25.0
    box(ax, qx, 53.3, qw, 4.6, "Post-alarm buffer (up to 10 s)")
    fw = (qw - 1.0) / 2
    for i, t in enumerate((r"$F_{td}$" "\ntime domain", r"$F_{cls}$" "\nCLS context",
                           r"$F_{att}$" "\nattention", r"$F_{graph}$" "\nlead graph")):
        r, c = divmod(i, 2)
        box(ax, qx + c * (fw + 1.0), 44.4 - r * 5.6, fw, 4.4, t, fs=6.8)
    box(ax, qx, 32.2, qw, 4.4, "Feature fusion and selection")
    box(ax, qx, 25.6, qw, 4.6, "SkEnsem\nstacked ensemble")
    box(ax, qx, 19.6, qw, 4.0, "NORM, MI, STTC, CD, HYP", weight="bold")
    down(ax, qx + qw / 2, 53.3, 48.8)
    down(ax, qx + qw / 2, 38.8, 36.6)
    down(ax, qx + qw / 2, 32.2, 30.2)
    down(ax, qx + qw / 2, 25.6, 23.6)
    elbow(ax, [(px + pw, 23.9), (68.5, 23.9), (68.5, 55.6), (qx, 55.6)])
    ax.text(69.3, 40.0, "alarm", fontsize=7.0, style="italic", color=NOTE, rotation=90, ha="center", va="center",
            zorder=4)

    # ---------------- contrastive change-point pretraining
    panel(ax, 1.0, 0.6, 98.0, 14.2, "Contrastive change-point pretraining (CCPP)")
    items = [("Anchor " r"$S_k$" "\n(recording " r"$a$" ")", 11.0),
             ("Positive " r"$S_k^{+}$" "\nneighbouring segment\nof the same recording", 15.5),
             ("Nuisance positive " r"$S_k^{u}$" "\nsplice to another patient\nwith the same labels", 15.5),
             ("Transition negative " r"$S_k^{-}$" "\nsplice to a recording\nwith other labels", 15.5),
             ("Post-change negative " r"$S_k^{b}$" "\nsegment of the new\nrecording", 15.5),
             ("Triplet loss over positives\nand negatives, with\nreconstruction and\ndiagnostic losses", 18.2)]
    x = 2.4
    for t, w in items:
        box(ax, x, 2.0, w, 8.4, t, fs=6.6)
        x += w + 0.8
    arrow(ax, 47.5, 14.8, 47.5, top, ls=(0, (2.5, 1.8)))
    ax.text(48.3, 16.15, "optimises HTFP and the encoder", fontsize=6.6, style="italic", color=NOTE, ha="left",
            va="center", zorder=4)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_framework.pdf")
    fig.savefig(out / "fig_framework.png", dpi=300)
    plt.close(fig)
    return out / "fig_framework.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "figures" / "generated"))
    args = ap.parse_args()
    print(framework(Path(args.out)))


if __name__ == "__main__":
    main()
