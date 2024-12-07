import pygame
import physics
import solids
from springer import Springer
from SpringerLogic import SpringerLogic

class SimulationManager:
    def __init__(self, display: pygame.Surface):
        self.display = display
        # self.lines = [physics.Line(physics.Point(200, 300), physics.Point(100, 400))]
        self.lines = [] # TODO Remove
        self.ground = solids.Ground(display)
        self.springers = [Springer(physics.Point(200, 300), 100, SpringerLogic())]

    def run_frame(self):
        self.run_physics()
        self.run_visual()

    def run_physics(self):
        # for line in self.lines:
            # line.fall()
            # physics.line_react_to_ground(line, self.ground.ground_line)
            # line.move()
        for springer in self.springers:
            springer.fall()
            springer.reactToGround(self.ground.ground_line)
            springer.move()
            springer.performAction(self.ground.ground_line)

    def run_visual(self):
        self.ground.draw()
        for springer in self.springers:
            pygame.draw.line(self.display, (255, 0, 0), springer.line.position_matrix[0], springer.line.position_matrix[1], 5)
        pygame.draw.line(self.display, (0, 0, 255), self.ground.ground_line.position_matrix[0], self.ground.ground_line.position_matrix[1], 5)
