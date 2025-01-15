import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(reward_list):
    plt.plot(reward_list, label='Rewards')
    x = np.arange(len(reward_list))  # X values (indices of reward_list)
    y = np.array(reward_list)        # Y values (reward values)
    slope, intercept = np.polyfit(x, y, 1)  # Linear regression (degree=1)
    y_fit = slope * x + intercept
    plt.plot(x, y_fit, color='red', label=f'Linear fit (slope={slope:.2f})')
    plt.grid()
    plt.legend()
    plt.show()