import numpy as np
import settings

class Point:
    def __init__(self, x: float, y: float):
        self.position_vector = np.array([x, y], dtype=np.float64)
        self.speed_vector = np.array([0, 0], dtype=np.float64)

    def __str__(self):
        return f"Pos: {str(self.position_vector)}, Speed: {str(self.speed_vector)}"
    
    def fall(self): # Not tested
        self.speed_vector[1] = self.speed_vector[1] - settings.settings["gravity"]

    def move(self):
        self.position_vector += self.speed_vector

def rotate_point_around_point(point_stationary: Point, point_rotating: Point, angle_rad: float):
    # Carries over speed of point without modifying
    relative_position = point_rotating.position_vector - point_stationary.position_vector
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_position = rotation_matrix @ relative_position
    final_rotated_position = rotated_position + point_stationary.position_vector

    new_position = Point(final_rotated_position[0], final_rotated_position[1])
    new_position.speed_vector = point_rotating.speed_vector[:]
    return new_position

def points_to_position_matrix(p1: Point, p2: Point):
    return np.array([p1.position_vector, p2.position_vector])

class Line:
    def __init__(self, p1: Point, p2: Point):
        self.position_matrix = np.array([p1.position_vector, p2.position_vector], dtype=np.float64)
        self.speed_matrix = np.array([[0, 0], [0, 0]], dtype=np.float64)
        self.length = distance(p1, p2)

    def p1(self):
        return Point(self.position_matrix[0][0], self.position_matrix[0][1])
    
    def p2(self):
        return Point(self.position_matrix[1][0], self.position_matrix[1][1])

    def move(self):
        self.position_matrix += self.speed_matrix

    def fall(self):
        self.speed_matrix[:, 1] += settings.settings["gravity"]

    def normalize(self):
        # Ignores speed

        # Move to p1 frame of reference
        self.position_matrix[1] -= self.position_matrix[0]
        self.position_matrix[0] -= self.position_matrix[0]
        
        # Cast to unit vector
        self.position_matrix[1] /= sum(self.position_matrix[1])
        self.length = 1

def position_matrix_to_line(matrix: np.array):
    return Line(Point(matrix[0][0], matrix[0][1]), Point(matrix[1][0], matrix[1][1]))

def angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0)) 

def rotate_line_around_point(point_stationary: Point, line: Line, angle: float):
    # Carries over speed of line without modifying
    p1_rotated = rotate_point_around_point(point_stationary, line.p1(), angle)
    p1_rotated.speed_vector = p1.speed_vector
    p2_rotated = rotate_point_around_point(point_stationary, line.p2(), angle)
    p2_rotated.speed_vector = p2.speed_vector
    return Line(p1_rotated, p2_rotated)

def line_react_to_ground(line: Line, ground: Line):
    # Assumes no actors outside visible plane
    if line.position_matrix[0][1] >= ground.position_matrix[0][1] and line.position_matrix[1][1] >= ground.position_matrix[0][1]:
        return "underground"
    intersect_point = intersect(line, ground)
    if intersect_point is None:
        # Ground too far away
        return "nothing"
    else:
        # Shift line to stay on the ground
        ground_point_index = 0 if line.position_matrix[0][1] > line.position_matrix[1][1] else 1
        offset = line.position_matrix[ground_point_index][1] - ground.position_matrix[0][1]
        line.position_matrix[:, 1] -= offset

        # Stop moving the bottom point
        line.speed_matrix[ground_point_index] = 0

        air_point_index = 1 - ground_point_index

        # Calculate unit vector towards ground point
        ground_unit_vec = line.position_matrix[ground_point_index] - line.position_matrix[air_point_index]
        ground_unit_vec /= np.sqrt((ground_unit_vec**2).sum())
        ground_unit_vec = Point(ground_unit_vec[0], ground_unit_vec[1])

        # Conserve only the perpendicular part
        perp_unit_vec = rotate_point_around_point(Point(0, 0), ground_unit_vec, np.pi/2)
        # TODO: Make the tie breaker below more reasonably written
        if ground_unit_vec.position_vector[0] == 0: # If leg 90 deg to ground then swing to the direction the speed is pointing to
            if line.speed_matrix[air_point_index][0] > 0:
                perp_unit_vec = Point(1, 0)
            else:
                perp_unit_vec = Point(-1, 0)
        if perp_unit_vec.position_vector[1] < 0: # If perp vector points up up then rotate it by 180 deg
            perp_unit_vec.position_vector *= -1
        speed_perp_angle = angle_between_vectors(perp_unit_vec.position_vector, line.speed_matrix[air_point_index])
        speed_module = np.sqrt(np.sum(line.speed_matrix[air_point_index]**2))
        line.speed_matrix[air_point_index] = perp_unit_vec.position_vector * abs(np.cos(speed_perp_angle)) * speed_module

        # a = line.position_matrix[0] - line.position_matrix[1]
        # print(np.sqrt(np.sum(a**2)))

        # Correct above ground point to maintain constant length # TODO: FINISH
        actual_lenght = np.sqrt(((line.position_matrix[air_point_index] - line.position_matrix[ground_point_index])**2).sum())
        correction_vector = (line.position_matrix[air_point_index] - line.position_matrix[ground_point_index]) * (line.length / actual_lenght - 1)
        line.position_matrix[air_point_index] += correction_vector

def intersect(l1: Line, l2: Line):
    p1, p2 = l1.position_matrix[0], l1.position_matrix[1]
    p3, p4 = l2.position_matrix[0], l2.position_matrix[1]
    
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
        if not (p1[0] <= intersection[0] <= p2[0] or p2[0] <= intersection[0] <= p1[0]) \
        or not (p1[1] <= intersection[1] <= p2[1] or p2[1] <= intersection[1] <= p1[1]):
            return None
        else:
            # Return intersection point when it's valid
            return Point(intersection[0], intersection[1])

    except np.linalg.LinAlgError:
        # Lines are parallel or coincident
        return None
    
def distance(p1: Point, p2: Point) -> float:
    return np.sqrt(((p1.position_vector - p2.position_vector)**2).sum())



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

    p1 = Point(0, 0)
    p2 = Point(1, 0)
    rotate_point_around_point(p1, p2, np.pi/2)
    print(p2) # Should be 0, 1