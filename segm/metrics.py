import numpy as np


def iou_pytorch(y_pred, y_true, eps=1e-6):
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum()

    return intersection / (union + eps)


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


def get_f1_score(preds, targets, threshold=0.5):
    f1 = []
    for p, t in zip(preds, targets):
        p_threshold = (p > threshold).float()
        f1.append(
            f1_score(p_threshold, t).item()
        )
    return np.mean(f1)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class IOUMetric:
    def __init__(self, class_index, threshold=0.5):
        self.class_index = class_index
        self.threshold = threshold
        self.avg_meter = AverageMeter()

    def __call__(self, preds, targets):
        preds_cls = preds[:, self.class_index]
        targets_cls = targets[:, self.class_index]
        iou = get_iou(preds_cls, targets_cls)
        self.avg_meter.update(iou, len(preds))

    def avg(self):
        return self.avg_meter.avg
