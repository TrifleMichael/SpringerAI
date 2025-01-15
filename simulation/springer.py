import physics
import numpy as np
from SpringerLogic import SpringerLogic
import settings
from random import randrange

class Springer:
    def __init__(self, starting_coords: physics.Point, length: float, logic: SpringerLogic):
        self.starting_coords = starting_coords
        self.line = physics.Line(starting_coords, physics.Point(starting_coords.position_vector[0] + randrange(-7, 8), starting_coords.position_vector[1]-length)) # TODO: Remove random twists
        self.logic = logic
        self.marked_for_removal = False # As objects cannot delete itself, this flag signals to simulation manager when a springer should be deleted

        self.step = 0
        self.state = {}
        self.last_jump = settings.settings["jump_cooldown"]
        self.last_shift = settings.settings["shift_cooldown"]
        self.speed = 0
        self.x_speed = 0

    # TODO: Remove ground line from update state
    def updateState(self, ground_line: physics.Line):
        self.state["step"] = self.step
        self.state["x_distance"] = max(self.line.position_matrix[0][0], self.line.position_matrix[1][0]) - self.starting_coords.position_vector[0]
        self.state["leg_angle"] = physics.angle_between_vectors(self.line.position_matrix[0] - self.line.position_matrix[1], np.array([1, 0]))
        # Leg angle is measured from bottom direction clockwise
        self.state["last_jump"] = self.last_jump
        self.state["last_shift"] = self.last_shift
        self.state["height"] = self.getHeight(ground_line)
        self.state["marked_for_removal"] = self.marked_for_removal
        self.state["speed"] = self.speed
        self.state["x_speed"] = self.x_speed
        leg_index = 1
        if self.line.position_matrix[0][1] > self.line.position_matrix[1][1]: # Leg has higher y (positive y is downwards)
            leg_index = 0

        self.state["x_leg_distance"] = self.line.position_matrix[leg_index][0] - self.starting_coords.position_vector[0]
        self.state["can_jump"] = 1 if self.getHeight(ground_line) == 0 and self.last_jump > settings.settings["jump_cooldown"] else 0

    def move(self):
        # Save some info about previous state
        previous_position = self.line.position_matrix.copy()

        self.line.move()

        # Update state variables after change
        self.calculateSpeed(previous_position)
        self.step += 1
        self.last_jump += 1
        self.last_shift += 1
    
    def fall(self):
        self.line.fall()

    def reactToGround(self, ground_line: physics.Line):
        result = physics.line_react_to_ground(self.line, ground_line)
        if result == "underground":
            self.marked_for_removal = True

    def getGroundIndex(self):
        if self.line.position_matrix[0][1] > self.line.position_matrix[1][1]:
            return 0
        else:
            return 1

    def calculateSpeed(self, previous_position: np.matrix):
        air_index = 1 - self.getGroundIndex()
        difference = self.line.position_matrix[air_index] - previous_position[air_index]
        self.speed = np.sqrt(sum(difference**2))
        self.x_speed = self.line.position_matrix[air_index][0] - previous_position[air_index][0]
        

    def getHeight(self, ground_line: physics.Line):
        if self.line.position_matrix[0][1] > self.line.position_matrix[1][1]:
            ground_index = 0
        else:
            ground_index = 1
        return ground_line.position_matrix[0][1] - self.line.position_matrix[ground_index][1]
    
    def getState(self):
        return self.state
    

    def performAction(self, ground_line: physics.Line):
        self.updateState(ground_line)
        action = self.logic.chooseAction(self.state)
        if action == "jump" and self.last_jump > settings.settings["jump_cooldown"]:
            self.last_jump = 0
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
                # print("Jumping")
                return action
        elif action in ["right", "left"]:
            if self.line.position_matrix[0][1] > self.line.position_matrix[1][1]:
                ground_index = 0
            else:
                ground_index = 1
            air_index = 1 - ground_index
            # Check if standing on the ground
            if self.line.position_matrix[ground_index][1] - ground_line.position_matrix[0][1] > -settings.settings["epsilon"]:
                if self.last_shift > settings.settings["shift_cooldown"]:
                    self.last_shift = 0
                    if action == "right":
                        # print("Moving right")
                        ground_head_vector = self.line.position_matrix[1 - air_index] - self.line.position_matrix[air_index]
                        ground_head_vector /= np.sqrt(sum(ground_head_vector**2))
                        perp_vector = np.array(ground_head_vector[1], -ground_head_vector[0]) # Rotate clockwise
                        self.line.speed_matrix[air_index] += perp_vector * settings.settings["side_force"]
                    elif action == "left":
                        # print("Moving left")
                        ground_head_vector = self.line.position_matrix[1 - air_index] - self.line.position_matrix[air_index]
                        ground_head_vector /= np.sqrt(sum(ground_head_vector**2))
                        perp_vector = np.array(-ground_head_vector[1], ground_head_vector[0]) # Rotate anticlockwise
                        self.line.speed_matrix[air_index] += perp_vector * settings.settings["side_force"]
        return action