import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 16})

# Color palette from the original study
STANDARD_RED = np.array([168, 14, 53]) / 255
BIOMIMETIC_BLUE = np.array([0, 82, 137]) / 255
ANTI_BIOMIMETIC_ORANGE = np.array([255, 165, 0]) / 255  # Orange for anti-biomimetic


def plot_fig_2_left(color_metrics, freq_metrics, regimen_name: str, az_metrics=None):
    """Create the Scatter Plot (Joint Distribution)

    Args:
        color_metrics: Array of color metric values
        freq_metrics: Array of frequency metric values
        regimen_name: Display name for the regimen (e.g., "Standard", "Biomimetic", or "Anti-biomimetic")
        az_metrics: Optional azimuth metrics (not used for coloring)
    """
    plt.figure(figsize=(8, 6))

    # Select color based on regimen name
    regimen_lower = regimen_name.lower()
    if "anti" in regimen_lower and (
        "biomimetic" in regimen_lower or "biomim" in regimen_lower
    ):
        point_color = ANTI_BIOMIMETIC_ORANGE
    elif "biomimetic" in regimen_lower or "biomim" in regimen_lower:
        point_color = BIOMIMETIC_BLUE
    else:
        point_color = STANDARD_RED  # Default to standard red

    plt.scatter(
        color_metrics,
        freq_metrics,
        color=point_color,  # Color based on regimen
        s=100,
        edgecolors="black",
        alpha=0.7,
    )

    # Colorbar removed - no longer coloring by azimuth index
    plt.xlabel("Color Metric")
    plt.ylabel("Frequency Metric")
    plt.title(f"Joint Distribution of RF Properties ({regimen_name})")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save and show
    # Generate filename based on regimen name
    regimen_safe = regimen_name.lower().replace(" ", "_").replace("-", "_")
    filename = f"{regimen_safe}_rf_distribution.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Plot saved as {filename}")
