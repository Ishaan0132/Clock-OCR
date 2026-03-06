import cv2
import numpy as np
import math
from .geometry_utils import distance_between_parallel_lines

def resize_image(img):
    height, width, _ = img.shape
    scale_factor = 1000 / max(height, width)
    img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
    return img

def clock_detection(img, blurred):
    radius = 0
    center_x, center_y = 0, 0

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 400, param1=50, param2=100, minRadius=100, maxRadius=500)
    max_circle = None

    if circles is not None:
        for circle in circles[0, :]:
            x, y, r = circle
            if r > radius:
                max_circle = circle

        if max_circle is not None:
            x, y, r = max_circle
            center_x, center_y, radius = int(x), int(y), int(r)
    else:
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_rect = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_rect = contour

        if max_rect is not None:
            (x, y, w, h) = cv2.boundingRect(max_rect)
            center_x = x + w // 2
            center_y = y + h // 2
            radius = min(w, h) // 2

    return center_x, center_y, radius

def line_detection(img, blurred):
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=30, maxLineGap=5)
    return lines

def group_lines_detection(lines, center_x, center_y, radius):
    groups = []
    if lines is None:
        return groups
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
        length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

        max_length = np.max([length1, length2])
        min_length = np.min([length1, length2])

        if (max_length < radius) and (min_length < radius * 50 / 100):
            angle = math.atan2(y2 - y1, x2 - x1)
            angle = math.degrees(angle)

            grouped = False
            for group in groups:
                mean_angle = group['mean_angle']
                if abs(angle - mean_angle) < 12 or abs(angle - mean_angle - 180) < 12 or abs(angle - mean_angle + 180) < 12:
                    group['lines'].append(line)
                    grouped = True
                    break

            if not grouped:
                groups.append({'lines': [line], 'mean_angle': angle})
    return groups

def hands_detection(groups, center_x, center_y):
    hands = []

    for group in groups:
        lines = group['lines']
        num_lines = len(lines)

        max_thickness = 0
        max_length = 0
        max_line = (0, 0, 0, 0) 

        for i in range(num_lines):
            x1, y1, x2, y2 = lines[i][0]
            length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
            length = np.max([length1, length2])

            if length > max_length:
                max_length = length
                if length == length1:
                    max_line = x1, y1, center_x, center_y
                else:
                    max_line = x2, y2, center_x, center_y

            for j in range(i+1, num_lines):
                thickness = distance_between_parallel_lines(lines[i], lines[j])
                if thickness > max_thickness:
                    max_thickness = thickness

        line = max_line, max_thickness, max_length
        if max_thickness > 0:
            hands.append(line)

    hands.sort(key=lambda x: x[2], reverse=True)
    hands = hands[:3]
    return hands

def get_hands(hands):
    if len(hands) < 3:
        raise ValueError("Could not detect all three hands properly.")

    sorted_hands_by_thickness = sorted(hands, key=lambda x: x[1])
    second_hand = sorted_hands_by_thickness[0]
    hands.remove(second_hand)

    sorted_hands_by_length = sorted(hands, key=lambda x: x[2])
    hour_hand = sorted_hands_by_length[0]
    minute_hand = sorted_hands_by_length[1]

    return hour_hand, minute_hand, second_hand