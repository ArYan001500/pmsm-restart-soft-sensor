"""Shared figure style for the manuscript (double-column journal layout).
Widths: single column 90 mm, 1.5 column 140 mm, full width 190 mm.  Vector PDF, TrueType fonts embedded (fonttype 42).
Palette: Okabe-Ito (CVD-safe, validated); every series also carries a distinct marker / line style (secondary encoding)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def latest_run(pattern):
    """Newest run directory in outputs/runs matching a glob pattern (run names start with a UTC time stamp)."""
    runs = sorted((ROOT / "outputs/runs").glob(pattern))
    if not runs:
        raise SystemExit(f"no run matching {pattern} in outputs/runs; run the corresponding experiment first")
    return runs[-1]
FIGDIR = ROOT / "outputs/figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM
C = {"CBP": "#0072B2", "naive_KF": "#D55E00", "oracle": "#009E73", "A1_labelprior": "#CC79A7", "A2_linear": "#E69F00",
     "A3_plainconf": "#56B4E9", "A4_uncal": "#8C8C8C", "A5_stdQ": "#6B4C9A", "A7_no_event_match": "#B07A00", "A8_simstart": "#4D4D4D",
     "A6_liang": "#555555", "init_winding": "#999999", "ink": "#1A1A1A", "muted": "#6E6E6E", "grid": "#E6E6E6", "band": "#0072B2"}
MK = {"CBP": "o", "naive_KF": "s", "oracle": "^", "A1_labelprior": "D", "A2_linear": "v", "A3_plainconf": "P", "A4_uncal": "X",
      "A5_stdQ": "h", "A7_no_event_match": "<", "A8_simstart": ">", "A6_liang": "*"}
LS = {"CBP": "-", "naive_KF": "--", "oracle": ":", "A1_labelprior": "-."}
LABEL = {"CBP": "CBP (proposed)", "naive_KF": "Naive KF", "oracle": "Oracle", "A1_labelprior": "A1 real-label prior",
         "A2_linear": "A2 linear prior", "A3_plainconf": "A3 no novelty", "A4_uncal": "A4 no conformal", "A5_stdQ": "A5 uniform Q",
         "A7_no_event_match": "A7 no event match", "A8_simstart": "A8 simulated start prior", "A6_liang": "Liang-type inversion"}
def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
        "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.edgecolor": "#333333", "axes.labelcolor": "#1A1A1A", "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150, "mathtext.fontset": "dejavusans"})
def panel(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right", color="#1A1A1A")
def save(fig, path):
    fig.savefig(str(path) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(path) + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
