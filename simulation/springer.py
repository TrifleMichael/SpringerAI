import physics
import numpy as np
from SpringerLogic import SpringerLogic
import settings

class Springer:
    def __init__(self, starting_coords: physics.Point, length: float, logic: SpringerLogic):
        self.line = physics.Line(starting_coords, physics.Point(starting_coords.position_vector[0], starting_coords.position_vector[1]-length))
        self.logic = logic

        self.step = 0

    def move(self):
        self.line.move()
        self.step += 1
    
    def fall(self):
        self.line.fall()

    def reactToGround(self, ground_line: physics.Line):
        physics.line_react_to_ground(self.line, ground_line)

    def performAction(self, ground_line: physics.Line):
        action = self.logic.chooseAction({"step": self.step})
        if action == "jump":
            print("JUMPING, at least trying")

            # TODO: Move to physics
            # Get ground and air index
            if self.line.position_matrix[0][1] > self.line.position_matrix[1][1]:
                ground_index = 0
            else:
                ground_index = 1
            air_index = 1 - ground_index
            # Check if standing on the ground
            if self.line.position_matrix[ground_index][1] - ground_line.position_matrix[0][1] > -settings.settings["epsilon"]:
                # Stop current movement
                self.line.speed_matrix *= 0                
                # Get unit vector for direction 
                unit_vec = self.line.position_matrix[air_index] - self.line.position_matrix[ground_index]
                unit_vec /= np.sqrt(np.sum(unit_vec**2))
                # Add multiplied unit tensor to speed
                self.line.speed_matrix += unit_vec * settings.settings["jump_force"]
                print("JUMP")
        elif action == "right":
            print("Moving right")
            if self.line.position_matrix[0][1] > self.line.position_matrix[1][1]:
                ground_index = 0
            else:
                ground_index = 1
            air_index = 1 - ground_index
            # Check if standing on the ground
            if self.line.position_matrix[ground_index][1] - ground_line.position_matrix[0][1] > -settings.settings["epsilon"]:
                self.line.speed_matrix[air_index][0] += settings.settings["side_force"]
            