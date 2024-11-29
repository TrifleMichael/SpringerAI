import pygame
import settings
import physics

class Ground:
    def __init__(self, display: pygame.Surface):
        self.display = display
        self.floor_height = settings.settings["y_res"] * (1 - settings.settings["ground_height_fraction"])
        self.ground_line = physics.Line(physics.Point(0, self.floor_height), physics.Point(settings.settings["x_res"], self.floor_height))

    def draw(self):
        height = settings.settings["y_res"] * settings.settings["ground_height_fraction"]
        width =  settings.settings["x_res"]
        pygame.draw.rect(self.display, (255, 255, 255), (0, self.floor_height, width, height))