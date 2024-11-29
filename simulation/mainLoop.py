import pygame
import sys
from math import radians
from simulationManager import SimulationManager
import settings

pygame.init()


# Screen dimensions
WIDTH, HEIGHT = settings.settings["x_res"], settings.settings["y_res"]
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rotated Rectangle")

sim_manager = SimulationManager(screen)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

rect_width, rect_height = 100, 50
rect_color = WHITE
rect_center = (WIDTH // 2, HEIGHT // 2)

angle = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # rect_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
    # rect_surface.fill(rect_color)

    # Rotate the rectangle
    # rotated_surface = pygame.transform.rotate(rect_surface, angle)
    # rotated_rect = rotated_surface.get_rect(center=rect_center)

    # Draw the rotated rectangle onto the screen
    # screen.blit(rotated_surface, rotated_rect.topleft)

    # angle += 1
    # if angle >= 360:
    #     angle -= 360

    sim_manager.run_frame()

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
