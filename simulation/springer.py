import physics
import numpy as np

class Springer:
    def __init__(self, starting_coords: physics.Point, length: float):
        self.starting_coords = starting_coords
        self.head = starting_coords
        self.bottom = np.array(self.head[0], self.head[1]+length)
        # Finish Line before continuing

        