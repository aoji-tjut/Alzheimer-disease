import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    # files = [ad_files, dementia_files, mci_files, normal_files, pd_files]

    ad = np.array([])
    dementia = np.array([])
    mci = np.array([])
    normal = np.array([])
    pd = np.array([])
    # nii = np.array([ad, dementia, mci, normal, pd])

    ####################################################################################################################
    for n_file in range(len(ad_files[2])):
        data = np.array(nib.load(str(ad_files[0]) + str(ad_files[2][n_file])).get_fdata())
        for n_channel in range(20):
            resize = cv.resize(data[:, :, n_channel], (256, 256))
            ad = np.append(ad, resize)
    ad = ad.reshape([-1, 256, 256, 20])
    print(ad.shape)

    for n_file in range(len(dementia_files[2])):
        data = np.array(nib.load(str(dementia_files[0]) + str(dementia_files[2][n_file])).get_fdata())
        for n_channel in range(20):
            resize = cv.resize(data[:, :, n_channel], (256, 256))
            dementia = np.append(dementia, resize)
    dementia = dementia.reshape([-1, 256, 256, 20])
    print(dementia.shape)

    for n_file in range(len(mci_files[2])):
        data = np.array(nib.load(str(mci_files[0]) + str(mci_files[2][n_file])).get_fdata())
        for n_channel in range(20):
            resize = cv.resize(data[:, :, n_channel], (256, 256))
            mci = np.append(mci, resize)
    mci = mci.reshape([-1, 256, 256, 20])
    print(mci.shape)

    for n_file in range(len(normal_files[2])):
        data = np.array(nib.load(str(normal_files[0]) + str(normal_files[2][n_file])).get_fdata())
        for n_channel in range(20):
            resize = cv.resize(data[:, :, n_channel], (256, 256))
            normal = np.append(normal, resize)
    normal = normal.reshape([-1, 256, 256, 20])
    print(normal.shape)

    for n_file in range(len(pd_files[2])):
        data = np.array(nib.load(str(pd_files[0]) + str(pd_files[2][n_file])).get_fdata())
        for n_channel in range(20):
            resize = cv.resize(data[:, :, n_channel], (256, 256))
            pd = np.append(pd, resize)
    pd = pd.reshape([-1, 256, 256, 20])
    print(pd.shape)

    X = np.vstack([normal, ad, dementia, mci, pd]).reshape(
        normal.shape[0] + ad.shape[0] + dementia.shape[0] + mci.shape[0] + pd.shape[0], -1)
    print(X.shape)
    ####################################################################################################################

    y = [0] * normal.shape[0] + [1] * ad.shape[0] + [2] * dementia.shape[0] + [3] * mci.shape[0] + [4] * pd.shape[0]
    y = np.asarray(y).reshape(-1, 1)
    print(y.shape)

    ####################################################################################################################
    np.savetxt("./data/X.txt", X)
    np.savetxt("./data/y.txt", y)


def LoadData():
    X = np.loadtxt("./data/X.txt")
    y = np.loadtxt("./data/y.txt")


    return X, y


def Preprocessing(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ss = StandardScaler()

    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    pickle.dump(ss, open("./ss.pkl", "wb"))

    X_train = X_train.reshape([-1, 256, 256, 20, 1])
    X_test = X_test.reshape([-1, 256, 256, 20, 1])

    pickle.dump(ss, open("./ss.pkl", "wb"))
    pickle.dump(ss, open("./ss.pkl", "wb"))
    pickle.dump(ss, open("./ss.pkl", "wb"))
    pickle.dump(ss, open("./ss.pkl", "wb"))


    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("Loading Data...")
    X, y = LoadData()

    print("Preprocessing Data...")
    X_train, X_test, y_train, y_test = Preprocessing(X, y)
    print("X_train.shape=", X_train.shape)
    print("y_train.shape=", y_train.shape)

    print("Training Data...")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv3D(64, kernel_size=5, strides=(3, 3, 1), padding="valid", activation="selu",
                                     input_shape=[256, 256, 20, 1]))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool3D())
    model.add(tf.keras.layers.Conv3D(128, kernel_size=3, strides=1, padding="valid", activation="selu"))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool3D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5, activation="softmax"))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["acc"])
    history = model.fit(X_train, y_train, batch_size=1, epochs=200, validation_split=0.2)

    print("Evaluate Data...")
    model.evaluate(X_test, y_test)

    print("Draw Figure...")
    plt.figure(1, (8, 5))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.plot(history.epoch, history.history.get("acc"), label="acc")
    plt.plot(history.epoch, history.history.get("val_acc"), label="val_acc")
    plt.legend()
    # plt.show()
    plt.savefig("./acc.png")

    plt.figure(2, (8, 5))
    plt.grid(True)
    plt.plot(history.epoch, history.history.get("loss"), label="loss")
    plt.plot(history.epoch, history.history.get("val_loss"), label="val_loss")
    plt.legend()
    # plt.show()
    plt.savefig("./loss.png")

    print("Save Model...")
    model.save("./model.h5")
