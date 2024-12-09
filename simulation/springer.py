import physics
import numpy as np
from SpringerLogic import SpringerLogic
import settings

class Springer:
    def __init__(self, starting_coords: physics.Point, length: float, logic: SpringerLogic):
        self.starting_coords = starting_coords
        self.line = physics.Line(starting_coords, physics.Point(starting_coords.position_vector[0], starting_coords.position_vector[1]-length))
        self.logic = logic
        self.marked_for_removal = False
        self.step = 0
        self.state = {}

    def updateState(self):
        self.state["step"] = self.step
        self.state["x_distance"] = max(self.line.position_matrix[0][0], self.line.position_matrix[1][0]) - self.starting_coords.position_vector[0]
        self.state["leg_angle"] = physics.angle_between_vectors(self.line.position_matrix[0] - self.line.position_matrix[1], np.array([0, 1]))
        # Leg angle is measured from bottom direction clockwise
        print(self.state["leg_angle"])

    def move(self):
        self.line.move()
        self.step += 1
    
    def fall(self):
        self.line.fall()

    def reactToGround(self, ground_line: physics.Line):
        result = physics.line_react_to_ground(self.line, ground_line)
        if result == "underground":
            self.marked_for_removal = True

    def performAction(self, ground_line: physics.Line):
        self.updateState()
        action = self.logic.chooseAction(self.state)
        if action == "jump":
            # TODO: Move to physics (jump(line, ground))
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
            