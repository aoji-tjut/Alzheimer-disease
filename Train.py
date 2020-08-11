import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nibabel as nib
import pickle


def LoadData():
    AD = []
    for i in range(15):
        image = nib.load("./samples/AD_nii/%d.nii" % (i + 1))
        image_array = np.array(image.get_fdata()).reshape(1, -1)
        AD = np.append(AD, image_array)
    AD = AD.reshape(15, -1)
    print(AD.shape)

    NC = []
    for i in range(15):
        image = nib.load("./samples/NC_nii/%d.nii" % (i + 1))
        image_array = np.array(image.get_fdata()).reshape(1, -1)
        NC = np.append(NC, image_array)
    NC = NC.reshape(15, -1)
    print(NC.shape)

    X = np.vstack([AD, NC])
    print(X.shape)

    y = [1] * 15 + [0] * 15
    y = np.asarray(y).reshape(30, 1)
    print(y.shape)

    return X, y


def Preprocessing(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ss = StandardScaler()

    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    pickle.dump(ss, open("./ss.pkl", "wb"))

    X_train = X_train.reshape([-1, 91, 109, 91, 1])
    X_test = X_test.reshape([-1, 91, 109, 91, 1])

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    # print(gpus, cpus)
    # tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
    # )

    X, y = LoadData()
    X_train, X_test, y_train, y_test = Preprocessing(X, y)
    print(X_train.shape)
    print(y_train.shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv3D(128, kernel_size=5, strides=3, padding="valid", activation="selu",
                                     input_shape=[91, 109, 91, 1]))
    model.add(tf.keras.layers.MaxPool3D())
    model.add(tf.keras.layers.Conv3D(256, kernel_size=3, strides=1, padding="valid", activation="selu"))
    model.add(tf.keras.layers.MaxPool3D())
    model.add(tf.keras.layers.Conv3D(512, kernel_size=3, strides=1, padding="valid", activation="selu"))
    model.add(tf.keras.layers.MaxPool3D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation="selu"))
    model.add(tf.keras.layers.Dense(256, activation="selu"))
    model.add(tf.keras.layers.Dense(32, activation="selu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003), loss="sparse_categorical_crossentropy",
                  metrics=["acc"])
    history = model.fit(X_train, y_train, batch_size=5, epochs=30, validation_split=0.2)

    model.evaluate(X_test, y_test)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    model.save("./model.h5")
