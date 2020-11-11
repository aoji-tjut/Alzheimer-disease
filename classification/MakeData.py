import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import nibabel as nib
import pickle
import cv2 as cv


def file_name(file_dir):
    for files in os.walk(file_dir):
        pass
    return files


def MakeData():
    ad_files = file_name("./data/AD/")
    dementia_files = file_name("./data/Dementia/")
    mci_files = file_name("./data/MCI/")
    normal_files = file_name("./data/Normal/")
    pd_files = file_name("./data/PD/")

    ad = np.array([])

    for n_file in range(len(ad_files[2])):
        data = np.array(nib.load(str(ad_files[0]) + str(ad_files[2][n_file])).get_fdata())
        data = data[50:-50, 50:-50, :]
        for n_channel in range(20):
            mms=MinMaxScaler((0,255))
            temp=mms.fit_transform(data[:,:,n_channel])
            resize = cv.resize(temp, (256, 256))
            ad = np.append(ad, resize)
    ad = ad.reshape([-1, 256, 256, 20])
    print(ad.shape)




if __name__ == '__main__':
    X = nib.load("I:\\hospital\\b311_enhance\\AD\\T1\\Train_01\\Train_01_t1.nii")
    X = np.array(X.get_fdata())
    mms = MinMaxScaler((0, 255))
    X = np.clip(X, 0, 255)
    X = mms.fit_transform(X[:, :, 8])
    plt.imshow(X, cmap="gray")

    X = cv.medianBlur(X, 3)
    plt.imshow(X, cmap="gray")
    X = np.array(X, np.uint8)
    plt.imshow(X, cmap="gray")
    X = cv.equalizeHist(X)
    plt.imshow(X, cmap="gray")

    plt.show()

    # MakeData()
