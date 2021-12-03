import numpy as np


def iou_pytorch(outputs, labels, eps=1e-6):
    intersection = (outputs * labels).sum()
    union = (outputs + labels).sum()

    return intersection / (union - intersection + eps)


def get_iou(preds, targets, threshold=0.5):
    iou = []
    for p, t in zip(preds, targets):
        p_threshold = (p > threshold).float()
        iou.append(
            iou_pytorch(p_threshold, t).item()
        )
    avg_iou = np.mean(iou)
    return avg_iou
