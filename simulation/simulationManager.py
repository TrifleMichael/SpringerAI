import pygame
import physics
import solids
from springer import Springer
from SpringerLogic import SpringerLogic

class SimulationManager:
    def __init__(self, display: pygame.Surface, springer_logic: SpringerLogic):
        self.display = display
        self.ground = solids.Ground(display)
        self.springers = [Springer(physics.Point(200, 300), 100, springer_logic)]

    def run_frame(self):
        self.run_physics()
        self.run_visual()

    def run_physics(self):
        for springer in self.springers:
            springer.fall()
            springer.reactToGround(self.ground.ground_line) # Must be after fall and before move (otherwise movement thround ground)
            springer.performAction(self.ground.ground_line) # Must be after react to ground and before move (jumps are negated at react to ground)
            springer.move()

        self.springers = [springer for springer in self.springers if not springer.marked_for_removal]

    def run_visual(self):
        self.ground.draw()
        for springer in self.springers:
            pygame.draw.line(self.display, (255, 0, 0), springer.line.position_matrix[0], springer.line.position_matrix[1], 5)
        pygame.draw.line(self.display, (0, 0, 255), self.ground.ground_line.position_matrix[0], self.ground.ground_line.position_matrix[1], 5)
