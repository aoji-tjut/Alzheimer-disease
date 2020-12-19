import cv2 as cv
import numpy as np
import nibabel as nib

data = np.array(nib.load("./data/Valid/AD/Valid_01/Valid_01_t2.nii").get_fdata())
data = data[:, :, 7]
mat = cv.getRotationMatrix2D((data.shape[1]*0.5,data.shape[0]*0.5),-90,1)
data = cv.warpAffine(data,mat,(data.shape[1],data.shape[0]))

scale = data.max() / 255
data = data / scale

cv.imwrite("../image/origin.png", data)
