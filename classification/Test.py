import tensorflow as tf
import numpy as np
import nibabel as nib
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def Paint(X, flag):
    X = ((X - X.min()) * (1 / (X.max() - X.min()) * 255)).astype('uint8')
    X = X.reshape(91, 109, 91)

    X1 = X[45, :, :]
    X2 = X[:, 55, :]
    X3 = X[:, :, 45]

    X1 = cv.cvtColor(X1, cv.COLOR_GRAY2RGB)
    X2 = cv.cvtColor(X2, cv.COLOR_GRAY2RGB)
    X3 = cv.cvtColor(X3, cv.COLOR_GRAY2RGB)
    X1 = cv.resize(X1, (X1.shape[1] * 3, X1.shape[0] * 3))
    X2 = cv.resize(X2, (X2.shape[1] * 3, X2.shape[0] * 3))
    X3 = cv.resize(X3, (X3.shape[1] * 3, X3.shape[0] * 3))

    B = (255, 0, 0)
    G = (0, 255, 0)
    R = (0, 0, 255)
    Y = (0, 255, 255)
    L = 2

    if flag == 1:
        cv.rectangle(X1, (59, 172), (85, 193), B, L)
        cv.rectangle(X2, (72, 82), (96, 119), B, L)


    elif flag == 2:
        cv.rectangle(X1, (63, 176), (91, 189), B, L)

        cv.rectangle(X2, (76, 76), (94, 112), B, L)
        cv.rectangle(X2, (100, 30), (125, 70), Y, L)

        cv.rectangle(X3, (93, 196), (135, 236), Y, L)

    else:
        cv.rectangle(X1, (62, 175), (83, 186), B, L)
        cv.rectangle(X1, (175, 226), (200, 250), R, L)
        cv.rectangle(X1, (157, 132), (173, 208), G, L)

        cv.rectangle(X2, (75, 80), (95, 115), B, L)
        cv.rectangle(X2, (100, 28), (135, 77), Y, L)
        cv.rectangle(X2, (156, 39), (182, 67), R, L)
        cv.rectangle(X2, (171, 119), (193, 151), G, L)

        cv.rectangle(X3, (100, 194), (133, 236), Y, L)
        cv.rectangle(X3, (278, 109), (297, 133), R, L)
        cv.rectangle(X3, (278, 142), (297, 165), R, L)
        cv.rectangle(X3, (231, 113), (271, 154), G, L)

    X1 = cv.resize(X1, (91, 109))
    X2 = cv.resize(X2, (91, 91))
    X3 = cv.resize(X3, (109, 91))

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.subplot(131)
    plt.imshow(X3)
    plt.title('轴向切面')
    plt.subplot(132)
    plt.imshow(X1)
    plt.title('矢状切面')
    plt.subplot(133)
    plt.imshow(X2)
    plt.title('冠状切面')

    color = ['lightskyblue', 'blue', 'red', 'green']
    labels = ['颞叶', '额叶', '海马体', '扣带回']
    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(handles=patches, loc='best', bbox_to_anchor=(2, 2), ncol=4)

    plt.savefig('./image/result/result.png', bbox_inches='tight', transparent=True)
    cv.imwrite("./image/result/X1_label.png", X1)
    cv.imwrite("./image/result/X2_label.png", X2)
    cv.imwrite("./image/result/X3_label.png", X3)


if __name__ == '__main__':
    X = nib.load("./samples/AD_nii/6.nii")  # test 2.nii 3.nii 6.nii 15.nii
    init = np.array(X.get_fdata())
    X = np.array(X.get_fdata()).reshape(1, -1)

    ss = pickle.load(open("./ss.pkl", "rb"))
    X = ss.transform(X)
    X = X.reshape([1, 91, 109, 91, 1])

    model = tf.keras.models.load_model("./model.h5")
    y_predict = model.predict(X)
    y = np.argmax(y_predict)

    if y:
        print("AD")
    else:
        print("NC")
    print("Probability = %f" % y_predict[0][y])

    if y:
        if y_predict[0][y] < 0.7:
            Paint(init, 1)
        elif y_predict[0][y] > 0.9:
            Paint(init, 3)
        else:
            Paint(init, 2)
