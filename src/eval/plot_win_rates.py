import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data
model_results = [
    {
        "method": "exponential",
        "weights": "w0.5",
        "harmless_win": 57.73,
        "harmless_tie": 6.19,
        "helpful_win": 51.03,
        "helpful_tie": 8.25,
    },
    {
        "method": "exponential",
        "weights": "risk2.0",
        "harmless_win": 58.76,
        "harmless_tie": 8.25,
        "helpful_win": 54.12,
        "helpful_tie": 6.70,
    },
    {
        "method": "exponential",
        "weights": "risk0.5",
        "harmless_win": 56.19,
        "harmless_tie": 7.73,
        "helpful_win": 52.06,
        "helpful_tie": 8.25,
    },
    {
        "method": "power",
        "weights": "w0.5",
        "harmless_win": 61.86,
        "harmless_tie": 4.64,
        "helpful_win": 45.36,
        "helpful_tie": 3.09,
    },
    {
        "method": "power",
        "weights": "risk0.2",
        "harmless_win": 61.34,
        "harmless_tie": 4.64,
        "helpful_win": 45.36,
        "helpful_tie": 3.09,
    },
    {
        "method": "power",
        "weights": "risk0.5",
        "harmless_win": 62.89,
        "harmless_tie": 4.12,
        "helpful_win": 45.88,
        "helpful_tie": 2.58,
    },
    {
        "method": "power",
        "weights": "risk0.8",
        "harmless_win": 61.34,
        "harmless_tie": 4.64,
        "helpful_win": 46.39,
        "helpful_tie": 3.09,
    },
    {
        "method": "chebyshev",
        "weights": "w0.5",
        "harmless_win": 54.12,
        "harmless_tie": 7.73,
        "helpful_win": 42.78,
        "helpful_tie": 4.64,
    },
    {
        "method": "linear",
        "weights": "w0.5",
        "harmless_win": 0.0,
        "harmless_tie": 100.0,
        "helpful_win": 0.0,
        "helpful_tie": 100.0,
    },
]

def plot_win_rates_scatter(output_dir: Path):
    """
    Create a scatter plot of harmlessness vs helpfulness win rates.
    Each method has its own color group and marker shape, with slight variations for different weights.
    Win rates are calculated as: win_rate + 0.5 * tie_rate
    
    Args:
        output_dir: Directory to save the plot
    """
    # Set up color palette and markers for methods
    method_styles = {
        'exponential': {'color': 'blue', 'marker': 'o'},
        'power': {'color': 'red', 'marker': 's'},
        'chebyshev': {'color': 'green', 'marker': '^'},
        'linear': {'color': 'purple', 'marker': 'D'}
    }
    
    # Create figure with white background
    plt.figure(figsize=(10, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Plot each point
    for result in model_results:
        method = result["method"]
        weights = result["weights"]
        
        # Calculate actual win rates (convert from percentage to decimal)
        harmless_rate = (result["harmless_win"] + 0.5 * result["harmless_tie"]) / 100
        helpful_rate = (result["helpful_win"] + 0.5 * result["helpful_tie"]) / 100
        
        # Get base style for method
        base_style = method_styles[method]
        
        # Create slightly different color for each weight
        if weights == "w0.5":
            color = base_style['color']
        elif weights == "risk0.2":
            color = plt.cm.get_cmap('Blues')(0.3) if method == 'exponential' else plt.cm.get_cmap('Reds')(0.3)
        elif weights == "risk0.5":
            color = plt.cm.get_cmap('Blues')(0.5) if method == 'exponential' else plt.cm.get_cmap('Reds')(0.5)
        elif weights == "risk0.8":
            color = plt.cm.get_cmap('Blues')(0.7) if method == 'exponential' else plt.cm.get_cmap('Reds')(0.7)
        elif weights == "risk2.0":
            color = plt.cm.get_cmap('Blues')(0.9) if method == 'exponential' else plt.cm.get_cmap('Reds')(0.9)
        
        # Plot point
        plt.scatter(
            harmless_rate,
            helpful_rate,
            color=color,
            marker=base_style['marker'],
            s=100,
            label=f"{method}_{weights}"
        )
    
    # Add reference lines
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Harmlessness Win Rate', fontsize=12)
    plt.ylabel('Helpfulness Win Rate', fontsize=12)
    plt.title('Win Rates Comparison', fontsize=14, pad=20)
    
    # Set axis limits
    plt.xlim(0.4, 0.7)
    plt.ylim(0.4, 0.7)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend with custom styling
    legend = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        edgecolor='black',
        fancybox=False
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "win_rates_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to {output_file}")

def main():
    """Generate all plots."""
    # Create output directory
    output_dir = Path("/Users/wan/Desktop/CS224R/project/output/advanced_modpo_infer/plots")
    output_dir.mkdir(exist_ok=True)
    
    # Generate scatter plot
    plot_win_rates_scatter(output_dir)

if __name__ == "__main__":
    main() 