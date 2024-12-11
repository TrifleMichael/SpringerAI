import pygame
import sys
from math import radians
from simulationManager import SimulationManager
from SpringerLogic import SpringerLogic
import settings

pygame.init()


# Screen dimensions
WIDTH, HEIGHT = settings.settings["x_res"], settings.settings["y_res"]
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SpringersAI")

springer_logic = SpringerLogic()
sim_manager = SimulationManager(screen, springer_logic)
sim_manager.spawn_springers(1)

running = True
generation = 0
while running:
    generations = int(input("How many generations should be ran? :"))
    run_animation = bool(input("Graphics mode for the chosen generations? y/n :") in ["y", "Y", "y\n", "Y\n"])

    # generations = 1
    # run_animation = True

    for generation in range(generations):
        sim_manager.simulate_generation(run_animation)

# Quit pygame
pygame.quit()
sys.exit()
