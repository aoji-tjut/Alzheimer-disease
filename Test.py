import tensorflow as tf
import numpy as np
import nibabel as nib
import pickle

X = nib.load("./samples/AD_nii/15.nii")
X = np.array(X.get_fdata()).reshape(1, -1)

ss = pickle.load(open("./ss.pkl", "rb"))
X = ss.transform(X)
X = X.reshape([1, 91, 109, 91, 1])

model = tf.keras.models.load_model("./model.h5")
y_predict = model.predict(X)
y = np.argmax(y_predict)
print(y_predict)
print(y)
#  1    2    4    5   6   7   8    9    10   11   12   13   14   15
# 9978 6754 9982 9985 1 9257 9988 9977 9988 9989 9983 9971 6995 7143