import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_style():
    # CJK-capable fonts on macOS / Linux / Windows, falling back silently
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS",
                                       "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
