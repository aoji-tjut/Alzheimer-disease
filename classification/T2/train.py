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


def LoadData():
    X = np.loadtxt("./X_T2.txt")
    y = np.loadtxt("./y_T2.txt")

    return X, y


def Preprocessing(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    pickle.dump(ss, open("./ss.pkl", "wb"))

    X_train = X_train.reshape([-1, 256, 256, 1])
    X_test = X_test.reshape([-1, 256, 256, 1])

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("Loading Data...")
    X, y = LoadData()

    print("Preprocessing Data...")
    X_train, X_test, y_train, y_test = Preprocessing(X, y)
    print("X_train.shape=", X_train.shape)
    print("y_train.shape=", y_train.shape)

    ####################################################################################################################
    print("Training Data...")
    # model = tf.keras.applications.ResNet50(include_top=False,pooling="max",classes=2)

    input = tf.keras.layers.Input(shape=[256, 256, 1])
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same")(input)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.layers.ReLU()(bn1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(relu1)

    # res1
    conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(pool1)
    bn2_1 = tf.keras.layers.BatchNormalization()(conv2_1)
    relu2_1 = tf.keras.layers.ReLU()(bn2_1)
    conv2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(relu2_1)
    bn2_2 = tf.keras.layers.BatchNormalization()(conv2_2)
    add = tf.keras.layers.add([pool1, bn2_2])
    relu2_2 = tf.keras.layers.ReLU()(add)

    conv2_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(relu2_2)
    bn2_3 = tf.keras.layers.BatchNormalization()(conv2_3)
    relu2_3 = tf.keras.layers.ReLU()(bn2_3)
    conv2_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(relu2_3)
    bn2_4 = tf.keras.layers.BatchNormalization()(conv2_4)
    add = tf.keras.layers.add([relu2_2, bn2_4])
    relu2_4 = tf.keras.layers.ReLU()(add)
    # 降维
    conv_temp = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=2, padding="valid")(relu2_4)

    # res2
    conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(relu2_4)
    bn3_1 = tf.keras.layers.BatchNormalization()(conv3_1)
    relu3_1 = tf.keras.layers.ReLU()(bn3_1)
    conv3_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(relu3_1)
    bn3_2 = tf.keras.layers.BatchNormalization()(conv3_2)
    add = tf.keras.layers.add([conv_temp, bn3_2])
    relu3_2 = tf.keras.layers.ReLU()(add)

    conv3_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(relu3_2)
    bn3_3 = tf.keras.layers.BatchNormalization()(conv3_3)
    relu3_3 = tf.keras.layers.ReLU()(bn3_3)
    conv3_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(relu3_3)
    bn3_4 = tf.keras.layers.BatchNormalization()(conv3_4)
    add = tf.keras.layers.add([relu3_2, bn3_4])
    relu3_4 = tf.keras.layers.ReLU()(add)
    # 降维
    conv_temp = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=2, padding="valid")(relu3_4)

    # res3
    conv4_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same")(relu3_4)
    bn4_1 = tf.keras.layers.BatchNormalization()(conv4_1)
    relu4_1 = tf.keras.layers.ReLU()(bn4_1)
    conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(relu4_1)
    bn4_2 = tf.keras.layers.BatchNormalization()(conv4_2)
    add = tf.keras.layers.add([conv_temp, bn4_2])
    relu4_2 = tf.keras.layers.ReLU()(add)

    conv4_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same")(relu4_2)
    bn4_3 = tf.keras.layers.BatchNormalization()(conv4_3)
    relu4_3 = tf.keras.layers.ReLU()(bn4_3)
    conv4_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu", padding="same")(relu4_3)
    bn4_4 = tf.keras.layers.BatchNormalization()(conv4_4)
    add = tf.keras.layers.add([relu4_2, bn4_4])
    relu4_4 = tf.keras.layers.ReLU()(add)
    # 降维
    conv_temp = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=2, activation="relu", padding="valid")(
        relu4_4)

    # res4
    conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")(relu4_4)
    bn5_1 = tf.keras.layers.BatchNormalization()(conv5_1)
    relu5_1 = tf.keras.layers.ReLU()(bn5_1)
    conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(relu5_1)
    bn5_2 = tf.keras.layers.BatchNormalization()(conv5_2)
    add = tf.keras.layers.add([conv_temp, bn5_2])
    relu5_2 = tf.keras.layers.ReLU()(add)

    conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(relu5_2)
    bn5_3 = tf.keras.layers.BatchNormalization()(conv5_3)
    relu5_3 = tf.keras.layers.ReLU()(bn5_3)
    conv5_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation="relu", padding="same")(relu5_3)
    bn5_4 = tf.keras.layers.BatchNormalization()(conv5_4)
    add = tf.keras.layers.add([relu5_2, bn5_4])
    relu5_4 = tf.keras.layers.ReLU()(add)
    pool2 = tf.keras.layers.AvgPool2D(pool_size=7, strides=1)(relu5_4)
    flatten = tf.keras.layers.Flatten()(pool2)
    # AvgPool2D(pool_size=height/width) + Flatten() = GlobalAvgPool2D() = [None, channels]

    # dense
    output = tf.keras.layers.Dense(2, activation="softmax")(flatten)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.summary()
    ####################################################################################################################

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["acc"])
    history = model.fit(X_train, y_train, batch_size=5, epochs=100, validation_split=0.2)

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
