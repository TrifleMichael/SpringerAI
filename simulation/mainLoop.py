import pygame
import sys
from math import radians
from EpsilonScheduler import EpsilonScheduler
from simulationManager import SimulationManager
from SpringerLogic import SpringerLogic, SpringerLogic_QLearning, SpringerLogic_QLearning_can_jump, SpringerLogic_Manual, SpringerLogic_QLearning_v2
import settings
import math
import numpy as np
from plotting import plot_rewards_and_scores

pygame.init()


# Screen dimensions
WIDTH, HEIGHT = settings.settings["x_res"], settings.settings["y_res"]
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SpringersAI")

EPSILON = EpsilonScheduler(
    settings.learing["epsilon_initial"], settings.learing["epsilon_target"], settings.learing["epsilon_decay"]
)

springer_logic = SpringerLogic_QLearning_v2(
    leg_angle_range=(0, math.pi),
    leg_angle_buckets=settings.learing["leg_angle_buckets"],
    learning_rate=settings.learing["learning_rate"],
    discount_factor=settings.learing["discount_factor"],
    epsilon=EPSILON,
)

# springer_logic = SpringerLogic_Manual()

sim_manager = SimulationManager(screen, springer_logic)
sim_manager.spawn_springers(1)

running = True
generation = 0
while running:
    generations = int(input("How many generations should be ran? :"))
    run_animation = bool(input("Graphics mode for the chosen generations? y/n :") in ["y", "Y", "y\n", "Y\n"])


    for generation in range(generations):
        print(f"--- Generation {generation+1} Epsilon {EPSILON.get_epsilon()}---")
        sim_manager.simulate_generation(run_animation)
        EPSILON.decay()

    if generations > 1:
        reward_list = settings.debug["reward_list"]
        score_list = settings.debug["score_list"]
        plot_rewards_and_scores(score_list, reward_list)

# Quit pygame
pygame.quit()
sys.exit()
