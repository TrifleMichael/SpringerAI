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

# Clock for controlling the frame rate
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

angle = 0

running = True
frame_counter = 0
while running:

    # Check if animation should be ran
    run_animation = settings.settings["first_animation_frame"] < frame_counter

    if run_animation:
        # Check for closing signal
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Clear frame
        screen.fill(BLACK)

    sim_manager.run_frame(run_animation)

    if run_animation:
        # Update the display
        pygame.display.flip()
        clock.tick(60)

    frame_counter += 1

# Quit pygame
pygame.quit()
sys.exit()
