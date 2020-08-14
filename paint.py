import numpy as np
import nibabel as nib
import cv2 as cv

X = nib.load("./samples/AD_nii/6.nii")
X = np.array(X.get_fdata())
print(X.shape)

X = ((X - X.min()) * (1 / (X.max() - X.min()) * 255)).astype('uint8')

X1 = X[45, :, :]
X2 = X[:, 55, :]
X3 = X[:, :, 45]

X1 = cv.cvtColor(X1, cv.COLOR_GRAY2RGB)
X2 = cv.cvtColor(X2, cv.COLOR_GRAY2RGB)
X3 = cv.cvtColor(X3, cv.COLOR_GRAY2RGB)

# cv.rectangle(X1, (20, 20), (40, 40), (0, 0, 255), 1)
# cv.rectangle(X2, (40, 40), (60, 60), (0, 0, 255), 1)
# cv.rectangle(X3, (60, 60), (80, 80), (0, 0, 255), 1)

cv.imwrite("./image/X1.png", X1)
cv.imwrite("./image/X2.png", X2)
cv.imwrite("./image/X3.png", X3)
