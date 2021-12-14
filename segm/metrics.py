import numpy as np


def iou_pytorch(y_pred, y_true, eps=1e-6):
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum()

    return intersection / (union - intersection + eps)


def get_iou(preds, targets, threshold=0.5):
    iou = []
    for p, t in zip(preds, targets):
        p_threshold = (p > threshold).float()
        iou.append(
            iou_pytorch(p_threshold, t).item()
        )
    return np.mean(iou)


def f1_score(y_pred, y_true, eps=1e-6):
    tp = (y_true * y_pred).sum()
    # tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision*recall / (precision + recall + eps)
    return f1


def get_f1_score(preds, targets, threshold):
    f1 = []
    for p, t in zip(preds, targets):
        p_threshold = (p > threshold).float()
        f1.append(
            f1_score(p_threshold, t).item()
        )
    return np.mean(f1)
