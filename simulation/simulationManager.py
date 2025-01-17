import pygame
import physics
import solids
from springer import Springer
from SpringerLogic import SpringerLogic
import settings
import copy

class SimulationManager:
    def __init__(self, display: pygame.Surface, springer_logic: SpringerLogic):
        self.display = display
        self.ground = solids.Ground(display)
        self.springer_logic = springer_logic
        self.default_spawning_coords = physics.Point(250, 500)
        self.default_springer_length = 100
        self.springers = []
        self.clock = pygame.time.Clock() # Clock for controlling the frame rate
        self.total_reward = 0
        self.delayed_learning_frames = 45

    def run_frame(self, run_animation: bool):
        self.run_physics()
        if run_animation:
            self.run_visual()

    def run_step(self, run_animation: bool):
        for springer in self.springers:            
            # Springer falls and reacts to the ground
            springer.fall()
            springer.reactToGround(self.ground.ground_line)

            springer.updateState(self.ground.ground_line)

            # Determine the action and save it
            if not springer.marked_for_removal:
                action = springer.performAction(self.ground.ground_line)
            else:
                action = None

            # Springer moves after reacting to the ground
            springer.move()

            springer.logic.update_knowledge(copy.deepcopy(springer.state), action)

            

        # Remove springers marked for removal
        self.springers = [springer for springer in self.springers if not springer.marked_for_removal]

        # Handle game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        # Run animation if enabled
        if run_animation:
            self.run_visual()

    def calculate_delayed_reward(self, springer, old_state, new_state):
        """
        Calculate the reward based on the delayed outcome of an action.
        Adjust this function to incorporate the logic for evaluating
        delayed results of actions.
        """
        return 1




    def run_physics(self):
        for springer in self.springers:
            springer.fall()
            springer.reactToGround(self.ground.ground_line) # Must be after fall and before move (otherwise movement thround ground)
            springer.performAction(self.ground.ground_line) # Must be after react to ground and before move (jumps are negated at react to ground, also springers must report their death that may have happened at reactToGround)
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
            self.run_step(run_animation)
            if len(self.springers) == 0 or iteration == settings.settings["frames_per_generation"] - 1:
                print(f"Generation ended after {iteration} iterations, final distance: {int(self.springer_logic.last_score)}")
                settings.debug["score_list"].append(self.springer_logic.last_score)
                settings.debug["reward_list"].append(self.springer_logic.total_rewards)
                print(f"q-table: {self.springer_logic.knowledge}")
                print(f"Total rewards:", self.springer_logic.total_rewards)
                self.springer_logic.total_rewards = 0
                break

            