import cv2
import numpy as np

def calculate_rect_coordinates(line):
    x1, y1, x2, y2 = line[0]
    rect_x = min(x1, x2)
    rect_y = min(y1, y2)
    rect_width = abs(x2 - x1)
    rect_height = abs(y2 - y1)
    text_x, text_y = x1, y1
    return rect_x, rect_y, rect_width, rect_height, text_x, text_y

def draw_hands_frame(img, hour_hand, minute_hand, second_hand):
    # Hour (Red)
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(hour_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 3)
    cv2.putText(img, 'Hour', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Minute (Green)
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(minute_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 3)
    cv2.putText(img, 'Minute', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Second (Blue)
    rect_x, rect_y, rect_width, rect_height, text_x, text_y = calculate_rect_coordinates(second_hand)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 3)
    cv2.putText(img, 'Second', (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def draw_time(img, time):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_color = (255, 255, 255)
    ribbon_color = (0, 0, 0)
    padding = 80

    h, w = img.shape[:2]

    # Create new image with space for ribbon
    new_img = np.ones((h + padding, w, 3), dtype=np.uint8) * 255
    new_img[0:h, :] = img

    cv2.rectangle(new_img, (0, h), (w, h + padding), ribbon_color, -1)

    (text_w, text_h), _ = cv2.getTextSize(time, font, font_scale, font_thickness)
    text_x = (w - text_w) // 2
    text_y = h + (padding + text_h) // 2

    cv2.putText(new_img, time, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return new_img