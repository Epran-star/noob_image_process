import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

rho = 1
theta = np.pi/180
threshold = 30
min_line_length = 50
max_line_gap = 20

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)

low_threshold = 100
high_threshold = 200
edges = cv2.Canny(gray, low_threshold, high_threshold)

line_image = np.copy(gray) #creating an image copy to draw lines on

# Run Hough on the edge-detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the image copy
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

plt.imshow(gray)
plt.imshow(line_image)
cv2.imwrite('hough.jpg',line_image)
