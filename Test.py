import tensorflow as tf
import numpy as np
import nibabel as nib
import pickle
import cv2 as cv

init = 0


def Paint(X, flag):
    X = ((X - X.min()) * (1 / (X.max() - X.min()) * 255)).astype('uint8')
    X=X.reshape(91,109,91)

    X1 = X[45, :, :]
    X2 = X[:, 55, :]
    X3 = X[:, :, 45]

    X1 = cv.cvtColor(X1, cv.COLOR_GRAY2RGB)
    X2 = cv.cvtColor(X2, cv.COLOR_GRAY2RGB)
    X3 = cv.cvtColor(X3, cv.COLOR_GRAY2RGB)
    print(X1.shape)

    B = (255, 0, 0)
    G = (0, 255, 0)
    R = (0, 0, 255)
    Y = (0, 255, 255)

    if flag == 1:
        cv.rectangle(X1, (int(0.422 * X1.shape[1]), int(0.692 * X1.shape[0])),
                     (int(0.483 * X1.shape[1]), int(0.795 * X1.shape[0])), B, 1)

        cv.rectangle(X2, (int(0.297 * X2.shape[1]), int(0.656 * X2.shape[0])),
                     (int(0.436 * X2.shape[1]), int(0.744 * X2.shape[0])), B, 1)
    elif flag == 2:
        cv.rectangle(X1, (int(0.428 * X1.shape[1]), int(0.678 * X1.shape[0])),
                     (int(0.471 * X1.shape[1]), int(0.784 * X1.shape[0])), B, 1)

        cv.rectangle(X2, (int(0.204 * X2.shape[1]), int(0.485 * X2.shape[0])),
                     (int(0.3 * X2.shape[1]), int(0.539 * X2.shape[0])), B, 1)
        cv.rectangle(X2, (int(0.083 * X2.shape[1]), int(0.405 * X2.shape[0])),
                     (int(0.188 * X2.shape[1]), int(0.469 * X2.shape[0])), Y, 1)

        cv.rectangle(X3, (int(0.714 * X3.shape[1]), int(0.596 * X3.shape[0])),
                     (int(0.868 * X3.shape[1]), int(0.725 * X3.shape[0])), Y, 1)
    else:
        cv.rectangle(X1, (int(0.443 * X1.shape[1]), int(0.703 * X1.shape[0])),
                     (int(0.474 * X1.shape[1]), int(0.78 * X1.shape[0])), B, 1)
        cv.rectangle(X1, (int(0.239 * X1.shape[1]), int(0.271 * X1.shape[0])),
                     (int(0.318 * X1.shape[1]), int(0.363 * X1.shape[0])), R, 1)
        cv.rectangle(X1, (int(0.373 * X1.shape[1]), int(0.374 * X1.shape[0])),
                     (int(0.612 * X1.shape[1]), int(0.44 * X1.shape[0])), G, 1)

        cv.rectangle(X2, (int(0.293 * X2.shape[1]), int(0.659 * X2.shape[0])),
                     (int(0.418 * X2.shape[1]), int(0.736 * X2.shape[0])), B, 1)
        cv.rectangle(X2, (int(0.103 * X2.shape[1]), int(0.418 * X2.shape[0])),
                     (int(0.278 * X2.shape[1]), int(0.637 * X2.shape[0])), Y, 1)
        cv.rectangle(X2, (int(0.139 * X2.shape[1]), int(0.337 * X2.shape[0])),
                     (int(0.245 * X2.shape[1]), int(0.436 * X2.shape[0])), R, 1)
        cv.rectangle(X2, (int(0.436 * X2.shape[1]), int(0.3 * X2.shape[0])),
                     (int(0.553 * X2.shape[1]), int(0.377 * X2.shape[0])), G, 1)

        cv.rectangle(X3, (int(0.711 * X3.shape[1]), int(0.599 * X3.shape[0])),
                     (int(0.861 * X3.shape[1]), int(0.697 * X3.shape[0])), Y, 1)
        cv.rectangle(X3, (int(0.399 * X3.shape[1]), int(0.101 * X3.shape[0])),
                     (int(0.484 * X3.shape[1]), int(0.159 * X3.shape[0])), R, 1)
        cv.rectangle(X3, (int(0.516 * X3.shape[1]), int(0.098 * X3.shape[0])),
                     (int(0.604 * X3.shape[1]), int(0.153 * X3.shape[0])), R, 1)
        cv.rectangle(X3, (int(0.414 * X3.shape[1]), int(0.177 * X3.shape[0])),
                     (int(0.568 * X3.shape[1]), int(0.3 * X3.shape[0])), G, 1)

    cv.imwrite("./image/X1_label.png", X1)
    cv.imwrite("./image/X2_label.png", X2)
    cv.imwrite("./image/X3_label.png", X3)


if __name__ == '__main__':
    X = nib.load("./samples/AD_nii/6.nii")
    init = np.array(X.get_fdata())
    X = np.array(X.get_fdata()).reshape(1, -1)

    ss = pickle.load(open("./ss.pkl", "rb"))
    X = ss.transform(X)
    X = X.reshape([1, 91, 109, 91, 1])

    model = tf.keras.models.load_model("./model.h5")
    y_predict = model.predict(X)
    y = np.argmax(y_predict)
    print(y_predict)
    print(y)
    Paint(X, 1)

    if y:
        if y_predict[0][y] < 0.7:
            Paint(init, 1)
        elif y_predict[0][y] > 0.9:
            Paint(init, 3)
        else:
            Paint(init, 2)
