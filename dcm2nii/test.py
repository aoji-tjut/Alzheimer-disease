import SimpleITK as sitk
from PIL import Image
import pydicom
import numpy as np
import cv2
import pprint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import nibabel as nib
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ds = sitk.ReadImage("./ADNI_011_S_4075_MR_MPRAGE_br_raw_20110610163917872_98_S111256_I239736.dcm")
# img_array = sitk.GetArrayFromImage(ds)
# frame_num, width, height = img_array.shape
# print(frame_num)
# print(width)
# print(height)
#
# information = {}
# ds = pydicom.read_file("./ADNI_011_S_4075_MR_MPRAGE_br_raw_20110610163917872_98_S111256_I239736.dcm")
# information['PatientID'] = ds.PatientID
# information['PatientName'] = ds.PatientName
# information['PatientBirthDate'] = ds.PatientBirthDate
# information['PatientSex'] = ds.PatientSex
# information['StudyID'] = ds.StudyID
# information['StudyDate'] = ds.StudyDate
# information['StudyTime'] = ds.StudyTime
# information['InstitutionName'] = ds.InstitutionName
# information['Manufacturer'] = ds.Manufacturer
# pprint.pprint(information)
#
# plt.imshow(img_array.reshape(256,240))
# plt.show()

# GetGDCMSeriesIDs读取序列号相同的dcm文件
series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs("./")
# GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames("./", series_id[0])
print(len(series_file_names))
series_reader = sitk.ImageSeriesReader()
series_reader.SetFileNames(series_file_names)
image3d = series_reader.Execute()
sitk.WriteImage(image3d, "./1.nii")

X = nib.load("./1.nii")
init = np.array(X.get_fdata())
X = np.array(X.get_fdata())
print(X.shape)
X1 = X[120, :, :]
X2 = X[:, 128, :]
X3 = X[:, :, 88]
plt.figure()
plt.imshow(X1)
plt.figure()
plt.imshow(X2)
plt.figure()
plt.imshow(X3)
plt.show()
