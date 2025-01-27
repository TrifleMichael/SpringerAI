import numpy as np
import settings
from numba import njit
import physics  # We assume physics is already Numba-optimized per your previous requests
from random import randrange

#
# 1. Numba-compiled helper functions for numeric operations
#

@njit
def _get_ground_index_numba(line_position_matrix):
    """
    Returns 0 if the first point is 'leg' (higher y), else returns 1.
    (Higher y means physically 'lower' if your coordinate system is inverted.)
    """
    if line_position_matrix[0, 1] > line_position_matrix[1, 1]:
        return 0
    else:
        return 1

@njit
def _calculate_speed_numba(previous_position, new_position, ground_index):
    """
    Calculates speed and x_speed based on movement of the 'air' index.
    """
    air_index = 1 - ground_index
    difference = new_position[air_index] - previous_position[air_index]
    speed = np.sqrt(difference[0]*difference[0] + difference[1]*difference[1])
    x_speed = new_position[air_index, 0] - previous_position[air_index, 0]
    return speed, x_speed

@njit
def _get_height_numba(line_position_matrix, ground_line_matrix, ground_index):
    """
    Returns how far above the ground the 'leg' point is.
    ground_line_matrix[0,1] is presumably the ground y-level.
    """
    return ground_line_matrix[0, 1] - line_position_matrix[ground_index, 1]

@njit
def _can_jump_numba(height, last_jump, jump_cooldown):
    """
    Checks if springer can jump: must be on ground (height=0)
    and last_jump > jump_cooldown.
    """
    if height == 0.0 and last_jump > jump_cooldown:
        return 1
    return 0

@njit
def _can_shift_numba(height, last_shift, shift_cooldown):
    """
    Checks if springer can shift: must be on ground (height=0)
    and last_shift >= shift_cooldown.
    """
    if height == 0.0 and last_shift >= shift_cooldown:
        return 1
    return 0

@njit
def _perform_jump_numba(line_position_matrix, line_speed_matrix, ground_index, jump_force):
    """
    Applies jump_force along the line direction (leg->air) to line_speed_matrix in-place.
    Resets speeds first (line_speed_matrix *= 0).
    """
    air_index = 1 - ground_index
    line_speed_matrix[:] = 0.0
    unit_vec = line_position_matrix[air_index] - line_position_matrix[ground_index]
    norm_uv = np.sqrt(np.sum(unit_vec*unit_vec))
    if norm_uv > 1e-15:
        unit_vec /= norm_uv
    line_speed_matrix += unit_vec * jump_force

@njit
def _perform_shift_numba(line_position_matrix, line_speed_matrix, air_index, ground_index,
                         side_force, direction):
    """
    Shifts the air point to the left or right of the 'leg->head' vector.
    direction is "right" or "left".
    """
    ground_head_vector = line_position_matrix[1 - air_index] - line_position_matrix[air_index]
    norm_ghv = np.sqrt(np.sum(ground_head_vector * ground_head_vector))
    if norm_ghv > 1e-15:
        ground_head_vector /= norm_ghv

    if direction == "right":
        # Clockwise 90 rotation
        perp_vector = np.array([ ground_head_vector[1], -ground_head_vector[0] ], dtype=np.float64)
    else:
        # Counterclockwise 90 rotation
        perp_vector = np.array([-ground_head_vector[1],  ground_head_vector[0] ], dtype=np.float64)

    line_speed_matrix[air_index] += perp_vector * side_force


#
# 2. The Springer class remains outwardly the same, but calls Numba helpers internally
#

class Springer:
    ACTIONS = ["jump", "left", "right", "pass"]

    def __init__(self, starting_coords: physics.Point, length: float, logic):
        self.starting_coords = starting_coords
        self.line = physics.Line(
            starting_coords,
            physics.Point(starting_coords.position_vector[0], starting_coords.position_vector[1] - length)
        )
        self.logic = logic
        self.marked_for_removal = False  # signals to simulation manager when a springer should be deleted

        self.step = 0
        self.state = {}
        self.last_jump = settings.settings["jump_cooldown"]
        self.last_shift = settings.settings["shift_cooldown"]
        self.speed = 0.0
        self.x_speed = 0.0
        self.pending_rewards = []
        self.state["dont_apply_penalty"] = False

    def updateState(self, ground_line: physics.Line):
        """
        Update self.state with numeric status. 
        """
        self.state["step"] = self.step
        self.state["x_distance"] = min(self.line.position_matrix[0][0], self.line.position_matrix[1][0]) \
                                   - self.starting_coords.position_vector[0]

        # Leg angle is measured from (line[0] -> line[1]) relative to [1, 0].
        # If your coordinate system is standard, you might want to define differently.
        vec = self.line.position_matrix[0] - self.line.position_matrix[1]
        self.state["leg_angle"] = physics.angle_between_vectors(vec, np.array([1.0, 0.0]))

        self.state["last_jump"] = self.last_jump
        self.state["last_shift"] = self.last_shift
        self.state["height"] = self.getHeight(ground_line)
        self.state["marked_for_removal"] = self.marked_for_removal
        self.state["speed"] = self.speed
        self.state["x_speed"] = self.x_speed

        # Leg index is whichever point has a larger y-value
        leg_index = 0 if self.line.position_matrix[0, 1] > self.line.position_matrix[1, 1] else 1
        self.state["x_leg_distance"] = self.line.position_matrix[leg_index, 0] - self.starting_coords.position_vector[0]

        # 1 => can jump/shift, 0 => cannot
        self.state["can_jump"] = _can_jump_numba(self.getHeight(ground_line), self.last_jump, settings.settings["jump_cooldown"])
        self.state["can_shift"] = _can_shift_numba(self.getHeight(ground_line), self.last_shift, settings.settings["shift_cooldown"])

    def move(self):
        """
        Moves the springer's line, updates step counters and speeds.
        """
        previous_position = self.line.position_matrix.copy()

        # Move line
        self.line.move()

        # Recompute speed
        self.calculateSpeed(previous_position)

        # Increment timers
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
        """
        Returns 0 if line[0] is the 'ground' foot, else 1 if line[1] is the foot.
        (Used for e.g. jump logic.)
        """
        return _get_ground_index_numba(self.line.position_matrix)

    def calculateSpeed(self, previous_position: np.ndarray):
        """
        Recompute self.speed, self.x_speed by comparing new vs old positions of 'air' index.
        """
        ground_index = self.getGroundIndex()
        new_position = self.line.position_matrix
        speed, x_speed = _calculate_speed_numba(previous_position, new_position, ground_index)
        self.speed = speed
        self.x_speed = x_speed

    def getHeight(self, ground_line: physics.Line):
        ground_index = self.getGroundIndex()
        return _get_height_numba(self.line.position_matrix, ground_line.position_matrix, ground_index)

    def getState(self):
        return self.state

    def performAction(self, ground_line: physics.Line):
        """
        Called each step to apply the chosen action from self.logic.
        """
        action = self.logic.chooseAction(self.state)

        if action == "jump" and self.last_jump > settings.settings["jump_cooldown"] and self.state["can_jump"]:
            ground_index = self.getGroundIndex()
            air_index = 1 - ground_index
            # Check if standing on ground
            # i.e. line[ground_index].y == ground_line[0].y
            if abs(self.line.position_matrix[ground_index, 1] - ground_line.position_matrix[0, 1]) < 1e-15:
                self.last_jump = 0
                _perform_jump_numba(
                    self.line.position_matrix,
                    self.line.speed_matrix,
                    ground_index,
                    settings.settings["jump_force"]
                )

        elif action in ["right", "left"] and self.state["can_shift"]:
            ground_index = self.getGroundIndex()
            air_index = 1 - ground_index
            # Check if standing on the ground
            if abs(self.line.position_matrix[ground_index, 1] - ground_line.position_matrix[0, 1]) < 1e-15:
                if self.last_shift >= settings.settings["shift_cooldown"]:
                    self.last_shift = 0
                    _perform_shift_numba(
                        self.line.position_matrix,
                        self.line.speed_matrix,
                        air_index,
                        ground_index,
                        settings.settings["side_force"],
                        action
                    )

        return action