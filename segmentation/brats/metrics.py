import numpy as np
import nibabel as nib

# 1-红-额叶
# 2-绿-颞叶
# 3-蓝-顶叶
# 4-黄-海马体
# 5-青-中脑
# 6-紫-半卵圆中心
y_label = nib.load("./data/Train/AD/Train_01/Train_01_seg.nii")
y_pred = nib.load("./ad_validation_predictions/Valid_01.nii")

y_label = np.array(y_label.get_fdata())
y_pred = np.array(y_pred.get_fdata())

# accuracy precision recall dice
for label in [1, 2, 3, 4, 5, 6]:
    iou = np.array([])
    accuracy = np.array([])
    precision = np.array([])
    recall = np.array([])
    for channel in range(y_label.shape[2]):
        two = 0             # y_pred正确的像素点
        one = 0             # y_label所有像素点
        tp = 0              # y_pred正样本 y_label正样本
        tn = 0              # y_pred负样本 y_label负样本
        fp = 0              # y_pred正样本 y_label负样本
        fn = 0              # y_pred负样本 y_label正样本

        for row in range(y_label.shape[0]):
            for col in range(y_label.shape[1]):
                # accuracy
                if y_label[row, col, channel] == label:
                    one += 1
                    if y_pred[row, col, channel] == label:
                        two += 1
                # confusion matrix
                if y_label[row, col, channel] == label and y_pred[row, col, channel] == label:
                    tp += 1
                elif y_label[row, col, channel] == label and y_pred[row, col, channel] != label:
                    fn += 1
                elif y_label[row, col, channel] != label and y_pred[row, col, channel] == label:
                    fp += 1
                else:
                    tn += 1
        if (fp + tp + fn) != 0 and tp != 0:
            iou = np.append(iou, tp / (fp + tp + fn))
        if one != 0 and two != 0:
            accuracy = np.append(accuracy, two / one)
        if (tp + fp) !=0  and tp != 0:
            precision = np.append(precision, (tp) / (tp + fp))
        if (tp + fn) != 0 and tp != 0:
            recall = np.append(recall, (tp) / (tp + fn))

    print("label\t\t", label)
    print(iou)
    print(accuracy)
    print(precision)
    print(recall)
    print("iou\t\t", iou.mean())
    print("accuracy\t", accuracy.mean())
    print("precision\t", precision.mean())
    print("recall\t\t", recall.mean())
    print("dice\t\t", (2 * precision.mean() * recall.mean()) / (precision.mean() + recall.mean()))
    print()
