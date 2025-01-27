import numpy as np
import settings
from numba import njit

#
# 1. Numba-compiled helper functions (operate on arrays, no Python objects)
#

@njit
def _angle_between_vectors_numba(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Numba helper for angle_between_vectors()."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-15 or norm2 < 1e-15:
        # Degenerate case; angle is undefined but we can return 0.0 as a safe default.
        return 0.0
    unit_vec1 = vec1 / norm1
    unit_vec2 = vec2 / norm2
    dot_product = np.dot(unit_vec1, unit_vec2)
    # Clip to avoid floating precision errors outside [-1, 1]
    dot_product = max(-1.0, min(1.0, dot_product))
    return np.arccos(dot_product)

@njit
def _rotate_point_around_point_numba(stationary: np.ndarray, rotating: np.ndarray, angle_rad: float) -> np.ndarray:
    """Numba helper for rotate_point_around_point()."""
    relative_position = rotating - stationary
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    # 2x2 rotation
    rotation_matrix = np.array([[c, -s],
                                [s,  c]], dtype=np.float64)
    rotated_position = rotation_matrix @ relative_position
    return rotated_position + stationary

@njit
def _distance_numba(p1: np.ndarray, p2: np.ndarray) -> float:
    """Numba helper for distance()."""
    diff = p1 - p2
    return np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])

@njit
def _intersect_numba(p1: np.ndarray, p2: np.ndarray,
                     p3: np.ndarray, p4: np.ndarray) -> np.ndarray or None:
    """Numba helper for intersect(). Returns np.ndarray or None."""
    d1 = p2 - p1
    d2 = p4 - p3

    # Build A matrix without np.array([d1, -d2]) to avoid Numba's "list of arrays" issue
    A = np.empty((2, 2), dtype=np.float64)
    A[:, 0] = d1
    A[:, 1] = -d2

    b = p3 - p1

    # Check determinant
    det = np.linalg.det(A)
    if abs(det) < 1e-15:
        # Lines are parallel or nearly so
        return None

    # Solve system
    solution = np.linalg.solve(A, b)
    t = solution[0]
    intersection = p1 + t * d1

    # Check intersection in-bounds
    if not (min(p1[0], p2[0]) <= intersection[0] <= max(p1[0], p2[0]) and
            min(p1[1], p2[1]) <= intersection[1] <= max(p1[1], p2[1])):
        return None
    return intersection

@njit
def _line_react_to_ground_check_numba(line_pos: np.ndarray, ground_pos: np.ndarray) -> int:
    """
    Numba helper to check line vs. ground intersection.
    Returns:
        0 -> 'underground'
        1 -> 'nothing'
        2 -> 'has intersection'
    """
    # If both points of line are above ground
    if (line_pos[0, 1] >= ground_pos[0, 1] and
        line_pos[1, 1] >= ground_pos[0, 1]):
        return 0  # 'underground'

    # Attempt intersection
    p1, p2 = line_pos[0], line_pos[1]
    p3, p4 = ground_pos[0], ground_pos[1]
    d1 = p2 - p1
    d2 = p4 - p3

    A = np.empty((2, 2), dtype=np.float64)
    A[:, 0] = d1
    A[:, 1] = -d2

    b = p3 - p1
    det = np.linalg.det(A)
    if abs(det) < 1e-15:
        return 1  # 'nothing' (parallel)

    solution = np.linalg.solve(A, b)
    t = solution[0]
    intersection = p1 + t * d1

    # Check if intersection is within bounds
    if not (min(p1[0], p2[0]) <= intersection[0] <= max(p1[0], p2[0]) and
            min(p1[1], p2[1]) <= intersection[1] <= max(p1[1], p2[1])):
        return 1  # 'nothing'

    return 2  # valid intersection => proceed to shifting logic in pure Python

#
# 2. Original classes and functions, calling the Numba helpers internally
#

class Point:
    def __init__(self, x: float, y: float):
        self.position_vector = np.array([x, y], dtype=np.float64)
        self.speed_vector = np.array([0.0, 0.0], dtype=np.float64)

    def __str__(self):
        return f"Pos: {str(self.position_vector)}, Speed: {str(self.speed_vector)}"
    
    def fall(self):
        """Apply gravity from settings."""
        self.speed_vector[1] -= settings.settings["gravity"]

    def move(self):
        """Move position by speed."""
        self.position_vector += self.speed_vector


def rotate_point_around_point(point_stationary: Point, point_rotating: Point, angle_rad: float):
    """
    Rotate point_rotating around point_stationary by angle_rad (radians).
    Carries over speed of point without modifying it.
    """
    rotated_pos = _rotate_point_around_point_numba(
        point_stationary.position_vector, point_rotating.position_vector, angle_rad
    )
    new_position = Point(rotated_pos[0], rotated_pos[1])
    new_position.speed_vector = point_rotating.speed_vector.copy()
    return new_position


def points_to_position_matrix(p1: Point, p2: Point):
    """Returns a 2x2 matrix with each point's position vector."""
    return np.array([p1.position_vector, p2.position_vector], dtype=np.float64)


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.position_matrix = np.array([p1.position_vector, p2.position_vector], dtype=np.float64)
        self.speed_matrix = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        self.length = distance(p1, p2)

    def p1(self):
        return Point(self.position_matrix[0, 0], self.position_matrix[0, 1])
    
    def p2(self):
        return Point(self.position_matrix[1, 0], self.position_matrix[1, 1])

    def move(self):
        self.position_matrix += self.speed_matrix

    def fall(self):
        self.speed_matrix[:, 1] += settings.settings["gravity"]

    def normalize(self):
        """
        Translate the line so that p1 is at origin,
        then convert p2 to a unit vector from p1.
        """
        self.position_matrix[1] -= self.position_matrix[0]
        self.position_matrix[0] -= self.position_matrix[0]
        # Cast to unit vector
        total = np.sum(self.position_matrix[1])
        if abs(total) > 1e-15:
            self.position_matrix[1] /= total
        self.length = 1.0


def position_matrix_to_line(matrix: np.ndarray):
    """
    Create a Line object from a 2x2 position matrix.
    """
    p1 = Point(matrix[0, 0], matrix[0, 1])
    p2 = Point(matrix[1, 0], matrix[1, 1])
    return Line(p1, p2)


def angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Returns the angle in radians between two vectors (range: 0 to pi).
    """
    return _angle_between_vectors_numba(vec1, vec2)


def rotate_line_around_point(point_stationary: Point, line: Line, angle: float):
    """
    Rotate an entire line around a stationary point by 'angle' radians.
    Carries over line's speed without modifying it.
    """
    p1_rot = rotate_point_around_point(point_stationary, line.p1(), angle)
    p2_rot = rotate_point_around_point(point_stationary, line.p2(), angle)

    # Preserve speed from the original line
    p1_rot.speed_vector = line.speed_matrix[0].copy()
    p2_rot.speed_vector = line.speed_matrix[1].copy()

    return Line(p1_rot, p2_rot)


def line_react_to_ground(line: Line, ground: Line):
    """
    Reaction of 'line' to 'ground'. 
    Returns one of:
       "underground"
       "nothing"
       None (if a valid intersection => we do final line shifting logic)
    """
    code = _line_react_to_ground_check_numba(line.position_matrix, ground.position_matrix)
    if code == 0:
        return "underground"
    elif code == 1:
        return "nothing"
    else:
        # We have a valid intersection => do final shifting logic
        ground_point_index = 0 if line.position_matrix[0, 1] > line.position_matrix[1, 1] else 1
        offset = line.position_matrix[ground_point_index, 1] - ground.position_matrix[0, 1]
        line.position_matrix[:, 1] -= offset

        # Stop moving the bottom point
        line.speed_matrix[ground_point_index] = 0

        air_point_index = 1 - ground_point_index

        # Calculate unit vector (ground -> air)
        ground_unit_vec = line.position_matrix[ground_point_index] - line.position_matrix[air_point_index]
        ground_unit_vec /= np.sqrt((ground_unit_vec ** 2).sum())

        # Choose rotation direction based on speed sign
        if line.speed_matrix[air_point_index, 0] < 0:
            perp_point = rotate_point_around_point(
                Point(0, 0),
                Point(ground_unit_vec[0], ground_unit_vec[1]),
                np.pi / 2
            )
            perp_unit_vec = perp_point.position_vector
        elif line.speed_matrix[air_point_index, 0] > 0:
            perp_point = rotate_point_around_point(
                Point(0, 0),
                Point(ground_unit_vec[0], ground_unit_vec[1]),
                -np.pi / 2
            )
            perp_unit_vec = perp_point.position_vector
        else:
            perp_unit_vec = None

        # Tie breaker if ground_unit_vec x == 0 => swing speed direction
        if abs(ground_unit_vec[0]) < 1e-15:
            if line.speed_matrix[air_point_index, 0] > 0:
                perp_unit_vec = np.array([1.0, 0.0], dtype=np.float64)
            else:
                perp_unit_vec = np.array([-1.0, 0.0], dtype=np.float64)

        if perp_unit_vec is not None:
            speed_perp_angle = angle_between_vectors(perp_unit_vec, line.speed_matrix[air_point_index])
            speed_module = np.linalg.norm(line.speed_matrix[air_point_index])
            line.speed_matrix[air_point_index] = perp_unit_vec * abs(np.cos(speed_perp_angle)) * speed_module

        # Correct above ground point to maintain constant length
        actual_length = np.linalg.norm(
            line.position_matrix[air_point_index] - line.position_matrix[ground_point_index]
        )
        correction_vector = (line.position_matrix[air_point_index] - line.position_matrix[ground_point_index]) \
                            * (line.length / actual_length - 1.0)
        line.position_matrix[air_point_index] += correction_vector
        return None


def intersect(l1: Line, l2: Line):
    """
    Returns a Point if there is an intersection, or None otherwise.
    """
    p1, p2 = l1.position_matrix[0], l1.position_matrix[1]
    p3, p4 = l2.position_matrix[0], l2.position_matrix[1]
    ipt = _intersect_numba(p1, p2, p3, p4)
    if ipt is None:
        return None
    else:
        return Point(ipt[0], ipt[1])


def distance(p1: Point, p2: Point) -> float:
    """Distance between two Points."""
    return _distance_numba(p1.position_vector, p2.position_vector)


# Testing
if __name__ == '__main__':
    # Provide a default 'settings'
    import sys
    class _Settings:
        def __init__(self):
            self.settings = {"gravity": 9.8}
    sys.modules['settings'] = _Settings()
    import settings  # reload with our class

    p1 = Point(0, 0)
    p2 = Point(1, 1)
    print(distance(p1, p2))  # Should be ~1.414 (sqrt(2))

    p3 = Point(0, 1)
    p4 = Point(1, 0)
    l1 = Line(p1, p2)
    l2 = Line(p3, p4)
    print(intersect(l1, l2))  # Should be near 0.5, 0.5

    p3 = Point(4, 5)
    p4 = Point(5, 4)
    l1 = Line(p1, p2)
    l2 = Line(p3, p4)
    print(intersect(l1, l2))  # Should be None

    p1 = Point(0, 0)
    p2 = Point(1, 0)
    new_p2 = rotate_point_around_point(p1, p2, np.pi / 2)
    print(new_p2)  # Should have position [0, 1]
