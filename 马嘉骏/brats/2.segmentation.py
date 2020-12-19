import cv2 as cv
import numpy as np
import nibabel as nib

data = cv.imread("../image/origin.png")
label = np.array(nib.load("./ad_validation_predictions/Valid_01.nii").get_fdata())
label = label[:, :, 7]
mat = cv.getRotationMatrix2D((label.shape[1] * 0.5, label.shape[0] * 0.5), -90, 1)
label = cv.warpAffine(label, mat, (label.shape[1], label.shape[0]))

for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        # BGR
        if label[row, col] == 1:  # 红
            data[row, col, 0] = 0
            data[row, col, 1] = 0
            data[row, col, 2] = 255
        if label[row, col] == 2:  # 绿
            data[row, col, 0] = 0
            data[row, col, 1] = 255
            data[row, col, 2] = 0
        if label[row, col] == 4:  # 黄
            data[row, col, 0] = 0
            data[row, col, 1] = 255
            data[row, col, 2] = 255
        if label[row, col] == 5:  # 青
            data[row, col, 0] = 255
            data[row, col, 1] = 255
            data[row, col, 2] = 0

cv.imwrite("../image/segmentation.png", data)
