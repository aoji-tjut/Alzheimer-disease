import tensorflow as tf
import numpy as np
import nibabel as nib
import pickle
import cv2 as cv


if __name__ == '__main__':
    X = nib.load("./ad.nii")
    init = np.array(X.get_fdata())
    X = np.array(X.get_fdata())

    high = X.shape[2]
    X = X[:, :, high // 2]
    X = cv.resize(X, (256, 256))
    X = X.reshape(1, -1)

    ss = pickle.load(open("./ss.pkl", "rb"))
    X = ss.transform(X)
    X = X.reshape([1, 256, 256, 1])

    model = tf.keras.models.load_model("./model.h5")
    y_predict = model.predict(X)
    y = np.argmax(y_predict)

    if y:
        print("AD")
    else:
        print("NC")
    print("Probability = %f" % y_predict[0][y])
