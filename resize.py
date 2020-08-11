import numpy as np
import cv2 as cv

X=cv.imread("./image/50-70/X1.png")
print(X.shape)
X=cv.resize(X,(X.shape[1]*3,X.shape[0]*3))
print(X.shape)
cv.imwrite("./image/50-70/X1.png",X)