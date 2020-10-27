import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nibabel as nib
import pickle

l,w,h=0,0,0

def file_name(file_dir):
    for files in os.walk(file_dir):
        pass
    return files

def LoadData():
    ad_files = file_name("./samples/AD_nii/")
    nc_files = file_name("./samples/NC_nii/")

    global l,w,h
    l,w,h=np.array(nib.load(str(ad_files[0]) + str(ad_files[2][0])).get_fdata()).shape
    print("l w h = ",l,w,h)

    AD = []
    num1=0
    for i in ad_files[2]:
        file = str(ad_files[0]) + str(ad_files[2][num1])
        print(file)
        image = nib.load(file)
        image_array = np.array(image.get_fdata())
        image_array = image_array[48:144,48:144,40:120]
        image_array = image_array.reshape(1, -1)
        AD = np.append(AD, image_array)
        num1=num1+1
    AD = AD.reshape(num1, -1)
    print("AD.shape=",AD.shape)

    NC = []
    num2 = 0
    for i in nc_files[2]:
        file=str(nc_files[0]) + str(nc_files[2][num2])
        print(file)
        image = nib.load(file)
        image_array = np.array(image.get_fdata())
        image_array = image_array[48:144, 48:144, 40:120]
        image_array = image_array.reshape(1, -1)
        NC = np.append(NC, image_array)
        num2=num2+1
    NC = NC.reshape(num2, -1)
    print("NC.shape=",NC.shape)

    X = np.vstack([AD, NC])
    print("X.shape=",X.shape)

    y = [1] * num1 + [0] * num2
    y = np.asarray(y).reshape(-1, 1)

    return X, y


def Preprocessing(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ss = StandardScaler()

    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    pickle.dump(ss, open("./ss.pkl", "wb"))

    X_train = X_train.reshape([-1, l//2, w//2, h//2, 1])
    X_test = X_test.reshape([-1, l//2, w//2, h//2, 1])

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X, y = LoadData()
    X_train, X_test, y_train, y_test = Preprocessing(X, y)
    print("X_train.shape=",X_train.shape)
    print("y_train.shape=",y_train.shape)

    with tf.device("/gpu:0"):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv3D(64, kernel_size=3, strides=1, padding="valid", activation="selu",
                                         kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                         input_shape=[l//2, w//2, h//2, 1]))
#        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.MaxPool3D())
        model.add(tf.keras.layers.Conv3D(128, kernel_size=3, strides=1, padding="valid", activation="selu",
                                         kernel_regularizer=tf.keras.regularizers.l2(0.1),))
#        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.MaxPool3D())
        model.add(tf.keras.layers.Conv3D(256, kernel_size=3, strides=1, padding="valid", activation="selu",
                                         kernel_regularizer=tf.keras.regularizers.l2(0.1),))
#        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.MaxPool3D())
        model.add(tf.keras.layers.Conv3D(512, kernel_size=3, strides=1, padding="valid", activation="selu",
                                         kernel_regularizer=tf.keras.regularizers.l2(0.1),))
#        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.MaxPool3D())
        model.add(tf.keras.layers.Flatten())
#        model.add(tf.keras.layers.Dense(1024, activation="selu"))
#        model.add(tf.keras.layers.Dropout(0.5))
#        model.add(tf.keras.layers.Dense(512, activation="selu"))
#        model.add(tf.keras.layers.Dropout(0.5))
#        model.add(tf.keras.layers.Dense(128, activation="selu"))
#        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(2, activation="softmax"))
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss="sparse_categorical_crossentropy",
                      metrics=["acc"])
        history = model.fit(X_train, y_train, batch_size=5, epochs=100, validation_split=0.2)

        model.evaluate(X_test, y_test)

    plt.figure(1,(8,5))
    plt.ylim(0,1)
    plt.grid(True)
    # plt.plot(history.epoch, history.history.get("loss"), label="loss")
    # plt.plot(history.epoch, history.history.get("val_loss"), label="val_loss")
    plt.plot(history.epoch, history.history.get("acc"), label="acc")
    plt.plot(history.epoch, history.history.get("val_acc"), label="val_acc")
    plt.legend()
    plt.show()
    plt.savefig("./a.png")

    model.save("./model.h5")
