import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 16})

# Color palette from the original study
STANDARD_RED = np.array([168, 14, 53]) / 255
BIOMIMETIC_BLUE = np.array([0, 82, 137]) / 255


def plot_fig_2_left(color_metrics, freq_metrics, az_metrics=None):
    """Create the Scatter Plot (Joint Distribution)"""
    plt.figure(figsize=(8, 6))
    plt.scatter(
        color_metrics,
        freq_metrics,
        color=STANDARD_RED,  # Single color for all points
        s=100,
        edgecolors="black",
        alpha=0.7,
    )

    # Colorbar removed - no longer coloring by azimuth index
    plt.xlabel("Color Metric")
    plt.ylabel("Frequency Metric")
    plt.title("Joint Distribution of RF Properties (Standard Regimen)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and show
    plt.savefig("standard_rf_distribution.png", dpi=300)
    plt.show()

    print("Plot saved as standard_rf_distribution.png")
