import pygame
import physics
import solids
from springer import Springer
from SpringerLogic import SpringerLogic
import settings

class SimulationManager:
    def __init__(self, display: pygame.Surface, springer_logic: SpringerLogic):
        self.display = display
        self.ground = solids.Ground(display)
        self.springer_logic = springer_logic
        self.default_spawning_coords = physics.Point(200, 300)
        self.default_springer_length = 100
        self.springers = []
        self.clock = pygame.time.Clock() # Clock for controlling the frame rate

    def run_frame(self, run_animation: bool):
        self.run_physics()
        if run_animation:
            self.run_visual()

    def run_physics(self):
        for springer in self.springers:
            springer.fall()
            springer.reactToGround(self.ground.ground_line) # Must be after fall and before move (otherwise movement thround ground)
            springer.performAction(self.ground.ground_line) # Must be after react to ground and before move (jumps are negated at react to ground)
            springer.move()

        self.springers = [springer for springer in self.springers if not springer.marked_for_removal]

        # Check for closing signal
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()


    def run_visual(self):
        # Clear frame
        self.display.fill((0, 0, 0))

        # Draw objects
        self.ground.draw()
        for springer in self.springers:
            pygame.draw.line(self.display, (255, 0, 0), springer.line.position_matrix[0], springer.line.position_matrix[1], 5)
        pygame.draw.line(self.display, (0, 0, 255), self.ground.ground_line.position_matrix[0], self.ground.ground_line.position_matrix[1], 5)
        
        # Update the display
        pygame.display.flip()
        self.clock.tick(60)

    def despawn_springers(self):
        self.springers = []

    def spawn_springers(self, number):
        self.springers += [Springer(self.default_spawning_coords, self.default_springer_length, self.springer_logic) for _ in range(number)]

    def simulate_generation(self, run_animation: bool):
        self.despawn_springers()
        self.spawn_springers(settings.settings["springers_per_generation"])
        for iteration in range(settings.settings["frames_per_generation"]):
            self.run_frame(run_animation)

            