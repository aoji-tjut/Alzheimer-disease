import tensorflow as tf
import numpy as np
import nibabel as nib
import pickle
import cv2 as cv

X1=cv.imread("./image/90-100/X1_.png")
X2=cv.imread("./image/90-100/X2_.png")
X3=cv.imread("./image/90-100/X3_.png")
print(X1.shape[0],X1.shape[1])
cv.imshow("X1",X1)
cv.imshow("X2",X2)
cv.imshow("X3",X3)
cv.waitKey(0)

# (shape[1],shape[0])
# X1(327,273) X2(273,273) X3(273,327)

# 50-70
# X1 blue (138,189) (158,217)
# X2 blue (81,179)  (119,203)

# 70-90
# X1 blue (140,185) (154,214)
# X2 blue (76,181)  (112,201)   yellow (31,151)  (70,175)
# X3 yellow (195,195) (237,237)

# 90-100
# X1 blue (145,192) (155,213)   red (78,74) (104,99)        green (122,102) (200,120)
# X2 blue (80,180)  (114,201)   yellow (28,114)  (76,174)   red (38,92) (67,119)        green (119,82) (151,103)
# X3 yellow (194,196) (235,228) red (109,33) (132,52) (141,32) (165,50)                 green (113,58) (155,98)