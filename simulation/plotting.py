import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    """Compute the moving average of a data sequence with a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_rewards_and_scores(score_list, reward_list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create two side-by-side subplots
    window_size = len(score_list) // 50  # Window size for moving average

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
    
    def plot_moving_metric(ax, data, title, label, window_size= 20):
        # Smooth the data using moving average
        smoothed_data = moving_average(data, window_size)
        
        ax.plot(smoothed_data, label=f"Smoothed {label} (Window={window_size})")
        x = np.arange(len(smoothed_data))
        y = np.array(smoothed_data)
        
        # Linear regression (degree=1) on smoothed data
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ax.plot(x, y_fit, color='red', label=f'Linear fit (slope={slope:.2f})')
        
        ax.set_title(title)
        ax.grid()
        ax.legend()


    # Plot rewards on the first subplot
    plot_moving_metric(axes[0], reward_list, "Reward sum", "Rewards", window_size)

    # Plot scores on the second subplot
    plot_moving_metric(axes[1], score_list, "Distance", "Distance", window_size)

    plt.tight_layout()
    plt.show()