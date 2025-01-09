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
        self.total_reward = 0

    def run_frame(self, run_animation: bool):
        self.run_physics()
        if run_animation:
            self.run_visual()

    def run_step(self, run_animation: bool):
        for springer in self.springers:
            # Save the old state before any updates
            old_state = springer.getState()
            
            # Springer falls and reacts to the ground
            springer.fall()
            springer.reactToGround(self.ground.ground_line)

            # Determine the action and save it
            action = springer.performAction(self.ground.ground_line)

            # Springer moves after reacting to the ground
            springer.move()

            # Mark springer for removal if it falls below the ground
            if springer.getState()["height"] < 0:
                springer.marked_for_removal = True

            # Update rewards if the action was valid
            if action is not None and not springer.marked_for_removal:
                reward = 1
                if action == "":
                    reward = 0.1

                self.total_reward += reward

                # Store the current action, state, and reward in history
                if not hasattr(springer, "action_history"):
                    springer.action_history = []
                springer.action_history.append((old_state.copy(), springer.getState().copy(), action, reward))
                
                # Limit history to a maximum length for efficiency
                if len(springer.action_history) > 3:  # Adjust the length as needed
                    springer.action_history.pop(0)

                # Update the knowledge with the immediate reward
                springer.logic.update_knowledge(old_state, action, reward, springer.getState())

            # Penalize recent actions if springer is marked for removal
            
            if springer.marked_for_removal:
                for idx, (old_state, new_state, action, reward) in enumerate(springer.action_history):
                    reward = -(idx+1)  # Define the penalty value
                    if action == "":
                        reward = -5
                    springer.logic.update_knowledge(old_state, action, reward, new_state)

        # Remove springers marked for removal
        self.springers = [springer for springer in self.springers if not springer.marked_for_removal]

        # Handle game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        # Run animation if enabled
        if run_animation:
            self.run_visual()



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
            if len(self.springers) == 0:
                print(f"Generation ended after {iteration} iterations, total reward: {self.total_reward}")
                print(f"q-table: {self.springer_logic.knowledge}")
                self.total_reward = 0
                break

            