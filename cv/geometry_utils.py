import numpy as np
import math

def distance_between_parallel_lines(line1, line2):
    """Calculates the perpendicular distance between two lines."""
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]

    vector1 = np.array([x2_1 - x1_1, y2_1 - y1_1])
    vector_between_lines = np.array([x1_2 - x1_1, y1_2 - y1_1])

    distance = np.abs(np.cross(vector1, vector_between_lines)) / np.linalg.norm(vector1)
    return distance

def dot_product(u, v):
    return u[0] * v[0] + u[1] * v[1]

def cross_product(u, v):
    return u[0] * v[1] - u[1] * v[0]

def get_vector(hand):
    x1, y1, x2, y2 = hand[0]
    return [x2 - x1, y2 - y1]