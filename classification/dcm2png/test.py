import numpy as np
import cv2
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import SimpleITK as sitk
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("./dcm")
reader.SetFileNames(dicom_names)
image = reader.Execute()
image_array = sitk.GetArrayFromImage(image)
img=np.array(image_array)

for num in range(img.shape[0]):
    temp=img[num,:,:]
    plt.imshow(temp)
    plt.savefig("./png/%d.png"%(num+1))


    # mms=MinMaxScaler((0,255))
    # temp=mms.fit_transform(temp)
    # cv.imwrite("./png/%d.png"%(num+1),temp)



