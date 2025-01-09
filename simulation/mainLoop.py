import pygame
import sys
from math import radians
from simulationManager import SimulationManager
from SpringerLogic import SpringerLogic, SpringerLogic_QLearning
import settings
import math
import numpy as np

pygame.init()


# Screen dimensions
WIDTH, HEIGHT = settings.settings["x_res"], settings.settings["y_res"]
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SpringersAI")

springer_logic = SpringerLogic_QLearning(   height_range=(-15, HEIGHT),  # Heights range from 0 to 2
                                            leg_angle_range=(0, math.pi),  # Leg angles range from 0 to 180 degrees
                                            height_buckets=6,
                                            leg_angle_buckets=7,
                                            epsilon=0.25,
                                            learning_rate=0.15)
# springer_logic.knowledge = { (0,0) : np.array([0,0,10,0]), (6,0) : np.array([0,10,0,0])}
sim_manager = SimulationManager(screen, springer_logic)
sim_manager.spawn_springers(1)

running = True
generation = 0
while running:
    generations = int(input("How many generations should be ran? :"))
    run_animation = bool(input("Graphics mode for the chosen generations? y/n :") in ["y", "Y", "y\n", "Y\n"])

    # run_animation = True

    for generation in range(generations):
        print(f"--- Generation {generation+1} ---")
        sim_manager.simulate_generation(run_animation)

# Quit pygame
pygame.quit()
sys.exit()
