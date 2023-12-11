import cv2
import numpy as np

def detect_corners_of_largest_square(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    sudoko_container = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(sudoko_container) == 4:
        for point in sudoko_container:
            cv2.circle(image, tuple(point[0]), 10, (0, 255, 0), -1)
    output_path = '4_Corners_Detection.jpg'
    cv2.imwrite(output_path, image)
    return output_path

detected_corners_image_path = detect_corners_of_largest_square('Testcases/16.jpg')
