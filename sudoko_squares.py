import cv2
import urllib
import numpy as np
import matplotlib.pyplot as plt


def detect_corners_of_largest_square(binary_image, mylist):
    # Apply a blur to reduce noise
    blur = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are found
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Process only if the approximated polygon has more than 4 points
        if len(approx_polygon) > 4:
            # Sort the points based on x-coordinate and then y-coordinate
            sorted_points = sorted(approx_polygon, key=lambda point: (point[0][0], point[0][1]))
            top_right = max(sorted_points, key=lambda point: point[0][0] - point[0][1])
            top_left = min(sorted_points, key=lambda point: point[0][0] + point[0][1])
            bottom_left = min(sorted_points, key=lambda point: point[0][0] - point[0][1])
            bottom_right = max(sorted_points, key=lambda point: point[0][0] + point[0][1])
            mylist.append(top_left)
            mylist.append(top_right)
            mylist.append(bottom_right)
            mylist.append(bottom_left)
            # Create a new list with the four corners
            corners = np.array([top_left, top_right, bottom_left, bottom_right])
        else:
            sorted_points = sorted(approx_polygon, key=lambda point: (point[0][0], point[0][1]))

            # Assuming the top-left corner will have the smallest sum of x and y,
            # and the bottom-right will have the largest sum.
            top_left = min(sorted_points, key=lambda point: point[0][0] + point[0][1])
            bottom_right = max(sorted_points, key=lambda point: point[0][0] + point[0][1])

            # The top-right corner will have the largest difference between x and y,
            # and the bottom-left will have the smallest difference.
            top_right = max(sorted_points, key=lambda point: point[0][0] - point[0][1])
            bottom_left = min(sorted_points, key=lambda point: point[0][0] - point[0][1])
            mylist.append(top_left)
            mylist.append(top_right)
            mylist.append(bottom_right)
            mylist.append(bottom_left)

            # Create a new list with the four corners
            corners = np.array([top_left, top_right, bottom_right, bottom_left])

        # Draw circles on the corners
        for point in corners:
            cv2.circle(binary_image, tuple(point[0]), 10, (255, 0, 0), -1)

        # Save or display the result
        return binary_image

mylist = []

#dilated image is the image output after all preprocessing

binary_image = detect_corners_of_largest_square(dilated_image , mylist)

for i in mylist:
  print(i)
plt.imshow(binary_image, cmap="gray");

p1 = mylist[0]
p2 = mylist[1]
p3 = mylist[2]
p4 = mylist[3]
tmp_img = np.zeros_like(binary_image, dtype = np.uint8)

coords = np.int32([[p1[::-1], p2[::-1], p3[::-1], p4[::-1]]])
tmp_img3 = np.zeros_like(binary_image, dtype = np.int32)
tmp_img3 = cv2.polylines(tmp_img3, coords, isClosed=True, color=(2550,0,0))
plt.imshow(tmp_img3 + binary_image, cmap="gray", vmax=1000);


y, x = binary_image.shape
src_coords = np.float32([[0,0], [x,0], [x,y], [0,y]])
dst_coords = np.float32([p1[::-1], p2[::-1], p3[::-1], p4[::-1]])
img_gray_threshed_warped = cv2.warpPerspective(
    src=binary_image,
    M=cv2.getPerspectiveTransform(dst_coords, src_coords),
    dsize=binary_image.shape[::-1]
)
plt.imshow(img_gray_threshed_warped, cmap="gray");


M = img_gray_threshed_warped.shape[0] // 9
N = img_gray_threshed_warped.shape[1] // 9
number_tiles = []
for i in range(9):
    number_tiles.append([])
    for j in range(9):
        tile = img_gray_threshed_warped[i*M:(i+1)*M, j*N:(j+1)*N]
        number_tiles[i].append(tile)

_, axes = plt.subplots(9, 9, figsize=(5, 5))
for i, row in enumerate(axes):
    for j, col in enumerate(row):
        col.imshow(number_tiles[i][j], cmap="gray");
        col.get_xaxis().set_visible(False)
        col.get_yaxis().set_visible(False)
