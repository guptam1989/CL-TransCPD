"""Figure style: fixed entity->colour mapping (validated reference palette, fixed slot order),
line styles/markers as secondary encoding for colour-vision deficiency and grayscale print."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# colour follows the entity in every figure
ENTITY = {
    "clt_full": SLOTS[0], "ocae": SLOTS[1], "tscp2": SLOTS[2], "tire": SLOTS[3], "lstm_ae": SLOTS[4],
    "bocpd": SLOTS[5], "rulsif": SLOTS[6], "cusum": SLOTS[7], "sup_resnet": INK2,   # supervised reference: neutral
    "prior_skensem": SLOTS[1], "resnet1d_wang": SLOTS[2], "inception1d": SLOTS[3], "transformer": SLOTS[4],
    "patchtst": SLOTS[5], "clocs": SLOTS[6], "lightgbm": SLOTS[7],
}
VARIANT_GRAY = "#898781"
LINESTYLE = {"clt_full": "-", "ocae": "--", "tscp2": "-.", "tire": ":", "lstm_ae": (0, (5, 1, 1, 1)),
             "bocpd": (0, (3, 1, 1, 1, 1, 1)), "rulsif": (0, (1, 1)), "cusum": (0, (6, 2)),
             "sup_resnet": (0, (8, 2, 2, 2))}
MARKER = {"clt_full": "o", "ocae": "s", "tscp2": "^", "tire": "D", "lstm_ae": "v", "bocpd": "P",
          "rulsif": "X", "cusum": "*", "sup_resnet": "h"}

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a32e2d"])


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        # sized so that full-width figures stay >= 6 pt when scaled to a 5.3-in preprint text width
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
