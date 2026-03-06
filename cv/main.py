import cv2
import os
from . import detection, time_calculation, visualization

def solve(img):
    # 1. Preprocessing
    img = detection.resize_image(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.bitwise_not(img_hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])

    _, thresh = cv2.threshold(img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # 2. Detection
    center_x, center_y, radius = detection.clock_detection(img, blurred)
    lines = detection.line_detection(img, blurred)
    groups = detection.group_lines_detection(lines, center_x, center_y, radius)
    hands = detection.hands_detection(groups, center_x, center_y)
    
    hour_hand, minute_hand, second_hand = detection.get_hands(hands)

    # 3. Visualization
    visualization.draw_hands_frame(img, hour_hand, minute_hand, second_hand)

    # 4. Time Calculation
    hour_angle = time_calculation.get_angle(hour_hand, center_x, center_y)
    minute_angle = time_calculation.get_angle(minute_hand, center_x, center_y)
    second_angle = time_calculation.get_angle(second_hand, center_x, center_y)

    time_str = time_calculation.get_time(hour_angle, minute_angle, second_angle)
    
    # 5. Final Output
    final_img = visualization.draw_time(img, time_str)

    return final_img, time_str

def main(img_path):
    # Check if file exists
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return

    img = cv2.imread(img_path)
    
    try:
        result_img, time_str = solve(img)
        
        print(f"Detected Time: {time_str}")

        # Show image
        cv2.imshow("Result", cv2.resize(result_img, (400, 400)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    img_path = r"images\clock4.jpg" 
    main(img_path)