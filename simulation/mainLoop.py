import pygame
import sys
from math import radians
from simulationManager import SimulationManager
from SpringerLogic import SpringerLogic, SpringerLogic_QLearning
import settings
import math
import numpy as np
from plotting import plot_rewards

pygame.init()


# Screen dimensions
WIDTH, HEIGHT = settings.settings["x_res"], settings.settings["y_res"]
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SpringersAI")

EPSILON = 1/400

springer_logic = SpringerLogic_QLearning(   height_range=(-1, 1.2),  # Heights range from 0 to 2
                                            leg_angle_range=(0, math.pi),  # Leg angles range from 0 to 180 degrees
                                            height_buckets=2,
                                            leg_angle_buckets=8,
                                            epsilon=EPSILON,
                                            learning_rate=0.1)
# springer_logic.knowledge = { (0,0) : np.array([0,0,10,0]), (6,0) : np.array([0,10,0,0])}
sim_manager = SimulationManager(screen, springer_logic)
sim_manager.spawn_springers(1)

running = True
generation = 0
while running:
    generations = int(input("How many generations should be ran? :"))
    run_animation = bool(input("Graphics mode for the chosen generations? y/n :") in ["y", "Y", "y\n", "Y\n"])
    # if generations < 10:
    #     springer_logic.epsilon = 0
    # else:
    springer_logic.epsilon = EPSILON

    # run_animation = True

    for generation in range(generations):
        print(f"--- Generation {generation+1} ---")
        sim_manager.simulate_generation(run_animation)

    reward_list = settings.debug["reward_list"]
    plot_rewards(reward_list)

# Quit pygame
pygame.quit()
sys.exit()
