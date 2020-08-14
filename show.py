import tensorflow as tf
import numpy as np
import nibabel as nib
import pickle
import cv2 as cv

X1=cv.imread("./image/90-100/X1_.png")
X2=cv.imread("./image/90-100/X2_.png")
X3=cv.imread("./image/90-100/X3_.png")
cv.imshow("X1",X1)
cv.imshow("X2",X2)
cv.imshow("X3",X3)
cv.waitKey(0)

