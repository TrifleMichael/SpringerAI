import pygame
import physics

class SimulationManager:
    def __init__(self, display: pygame.Surface):
        self.display = display
        lines = []

    def run_frame(self):
        for line in self.lines:
            pass # Draw a line in pygame
