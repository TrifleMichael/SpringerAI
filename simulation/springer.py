import physics as ph
import numpy as np

class Springer:
    def __init__(self, starting_coords: ph.Point, length: float):
        self.starting_coords = starting_coords
        self.head = starting_coords
        self.bottom = np.array(self.head[0], self.head[1]-length)

        