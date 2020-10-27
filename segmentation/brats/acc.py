import numpy as np
import nibabel as nib

# 1-红-额叶
# 2-绿-颞叶
# 3-蓝-顶叶
# 4-黄-海马体
# 5-青-中脑
# 6-紫-半卵圆中心
y_label = nib.load("C:/Users/Administrator/Desktop/result/seg.nii")
# y_pred = nib.load("C:/Users/Administrator/Desktop/result/pre_64-64-20_32-32-32_16_0.0003_0.5.nii")
y_pred = nib.load("C:/Users/Administrator/Desktop/Valid_01.nii")

# BraTS2020
# y_label = nib.load("C:/Users/Administrator/Desktop/result/BraTS2020/seg.nii")
# y_pred = nib.load("C:/Users/Administrator/Desktop/result/BraTS2020/pre.nii")

y_label = np.array(y_label.get_fdata())
y_pred = np.array(y_pred.get_fdata())

# accuracy precision recall dice
for label in [1, 2, 3, 4, 5, 6]:
    iou = np.array([])
    accuracy = np.array([])
    precision = np.array([])
    recall = np.array([])
    for channel in range(y_label.shape[2]):
        intersection = 0
        union = 0
        tp = 0
        fn = 0
        fp = 0
        tn = 0

        for row in range(y_label.shape[0]):
            for col in range(y_label.shape[1]):
                # IoU
                if (y_label[row, col, channel] == label or y_pred[row, col, channel] == label):
                    union += 1
                if (y_label[row, col, channel] == label and y_pred[row, col, channel] == label):
                    intersection += 1
                # confusion matrix
                if y_label[row, col, channel] == label and y_pred[row, col, channel] == label:
                    tp += 1
                elif y_label[row, col, channel] == label and y_pred[row, col, channel] != label:
                    fn += 1
                elif y_label[row, col, channel] != label and y_pred[row, col, channel] == label:
                    fp += 1
                else:
                    tn += 1

        if union != 0:
            iou = np.append(iou, intersection / union)
        if (tp + fn + fp + tn) != 0:
            accuracy = np.append(accuracy, (tp + tn) / (tp + fn + fp + tn))
        if (tp + fp) != 0:
            precision = np.append(precision, (tp) / (tp + fp))
        if (tp + fn) != 0:
            recall = np.append(recall, (tp) / (tp + fn))

    print("label\t\t", label)
    print("iou\t\t\t", iou.mean())
    print("accuracy\t", accuracy.mean())
    print("precision\t", precision.mean())
    print("recall\t\t", recall.mean())
    print("dice\t\t", (2 * precision.mean() * recall.mean()) / (precision.mean() + recall.mean()))
    print()
