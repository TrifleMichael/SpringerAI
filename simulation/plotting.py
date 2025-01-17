import matplotlib.pyplot as plt
import numpy as np

def plot_rewards_and_scores(score_list, reward_list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create two side-by-side subplots

    # Function to plot a single metric with linear fit
    def plot_metric(ax, data, title, label):
        ax.plot(data, label=label)
        x = np.arange(len(data))
        y = np.array(data)
        slope, intercept = np.polyfit(x, y, 1)  # Linear regression (degree=1)
        y_fit = slope * x + intercept
        ax.plot(x, y_fit, color='red', label=f'Linear fit (slope={slope:.2f})')
        ax.set_title(title)
        ax.grid()
        ax.legend()

    # Plot rewards on the first subplot
    plot_metric(axes[0], reward_list, "Reward sum", "Rewards")

    # Plot scores on the second subplot
    plot_metric(axes[1], score_list, "Distance", "Distance")

    plt.tight_layout()
    plt.show()