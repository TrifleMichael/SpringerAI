import pygame
import physics
import solids

class SimulationManager:
    def __init__(self, display: pygame.Surface):
        self.display = display
        self.lines = [physics.Line(physics.Point(120, 300), physics.Point(100, 400))]
        self.ground = solids.Ground(display)

    def run_frame(self):
        self.run_physics()
        self.run_visual()

    def run_physics(self):
        for line in self.lines:
            line.fall()
            physics.line_react_to_ground(line, self.ground.ground_line)
            line.move()

    def run_visual(self):
        self.ground.draw()
        for line in self.lines:
            pygame.draw.line(self.display, (255, 0, 0), line.position_matrix[0], line.position_matrix[1], 5)
        pygame.draw.line(self.display, (0, 0, 255), self.ground.ground_line.position_matrix[0], self.ground.ground_line.position_matrix[1], 5)
