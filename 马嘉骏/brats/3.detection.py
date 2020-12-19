import cv2 as cv
import numpy as np
import nibabel as nib

data = cv.imread("../image/origin.png")
cv.rectangle(data, (170, 140), (240, 180), (0, 0, 255), 3)
cv.rectangle(data, (280, 140), (360, 180), (0, 0, 255), 3)
cv.rectangle(data, (130, 190), (200, 440), (0, 255, 0), 3)
cv.rectangle(data, (330, 190), (390, 440), (0, 255, 0), 3)
cv.rectangle(data, (190, 230), (230, 340), (0, 255, 255), 3)
cv.rectangle(data, (300, 230), (340, 350), (0, 255, 255), 3)
cv.rectangle(data, (230, 250), (300, 310), (255, 255, 0), 3)

cv.imwrite("../image/detection.png", data)
