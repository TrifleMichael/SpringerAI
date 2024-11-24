import numpy as np
import settings

# TODO:
# Add visualisation
# Add gravity (things must fall)
# Add rotation (lines must spin)
# Angular momentum might hurt
# Add springers

class Point:
    def __init__(self, x: float, y: float):
        self.arr = np.array([x, y])
        self.speed = np.array([0, 0])

    def __str__(self):
        return f"Pos: {str(self.arr)}, Speed: {str(self.speed)}"
    
    def fall(self): # Not tested
        self.speed[1] = self.speed[1] - settings.settings["gravity"]

class Line:
    def __init__(self, p1: Point, p2: Point):
        self.matrix = np.array([p1.arr, p2.arr])

def intersect(l1: Line, l2: Line):
    p1, p2 = l1.matrix[0], l1.matrix[1]
    p3, p4 = l2.matrix[0], l2.matrix[1]
    
    # Direction vectors for each line
    d1 = p2 - p1
    d2 = p4 - p3
    
    # Solve the linear system
    A = np.array([d1, -d2]).T  # Coefficients matrix
    b = p3 - p1  # Right-hand side

    try:
        # Solve for t and u
        t, u = np.linalg.solve(A, b)
        
        # Intersection point
        intersection = p1 + t * d1

        # Check if intersection is out of bounds
        if not (p1[0] < intersection[0] < p2[0] or p2[0] < intersection[0] < p1[0]) \
        or not (p1[1] < intersection[1] < p2[1] or p2[1] < intersection[1] < p1[1]):
            return None
        else:
            return Point(intersection[0], intersection[1])

    except np.linalg.LinAlgError:
        # Lines are parallel or coincident
        return None

def distance(p1: Point, p2: Point) -> float:
    return np.sqrt(((p1.arr - p2.arr)**2).sum())




# Testing
if __name__ == '__main__':
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    print(distance(p1, p2)) # Should be sqrt(2)

    p3 = Point(0, 1)
    p4 = Point(1, 0)
    l1 = Line(p1, p2)
    l2 = Line(p3, p4)
    print(intersect(l1, l2)) # Should be 0.5 0.5

    p3 = Point(4, 5)
    p4 = Point(5, 4)
    l1 = Line(p1, p2)
    l2 = Line(p3, p4)
    print(intersect(l1, l2)) # Should be None

