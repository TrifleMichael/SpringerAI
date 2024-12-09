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
pygame.display.set_caption("Rotated Rectangle")

springer_logic = SpringerLogic()
sim_manager = SimulationManager(screen, springer_logic)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

angle = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    sim_manager.run_frame()

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
