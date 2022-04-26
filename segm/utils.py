import torch
import os
import math
import logging
import time
from tqdm import tqdm

from segm.metrics import get_iou, get_f1_score, AverageMeter, IOUMetric
from segm.predictor import predict


def configure_logging(log_path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    # Setup console logging
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    # Setup file logging as well
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def val_loop(data_loader, model, criterion, device, class_names, logger):
    loss_avg = AverageMeter()
    iou_avg = AverageMeter()
    cls2iou = {cls_name: IOUMetric(cls_idx)
               for cls_idx, cls_name in enumerate(class_names)}
    f1_score_avg = AverageMeter()
    strat_time = time.time()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, targets in tqdm_data_loader:
        preds, targets = predict(images, model, device, targets)
        batch_size = len(images)

        loss = criterion(preds, targets)
        loss_avg.update(loss.item(), batch_size)

        iou_avg.update(get_iou(preds, targets), batch_size)
        f1_score_avg.update(get_f1_score(preds, targets), batch_size)
        for cls_name in class_names:
            cls2iou[cls_name](preds, targets)
    loop_time = sec2min(time.time() - strat_time)
    cls2iou_log = ''.join([f' IOU {cls_name}: {iou_fun.avg():.4f}'
                           for cls_name, iou_fun in cls2iou.items()])
    logger.info(f'Validation, '
                f'Loss: {loss_avg.avg:.4f}, '
                f'IOU avg: {iou_avg.avg:.4f}, '
                f'{cls2iou_log}, '
                f'F1 avg: {f1_score_avg.avg:.4f}, '
                f'loop_time: {loop_time}')
    return loss_avg.avg


def sec2min(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class FilesLimitControl:
    """Delete files from the disk if there are more files than the set limit.
    Args:
        max_weights_to_save (int, optional): The number of files that will be
            stored on the disk at the same time. Default is 3.
    """
    def __init__(self, logger=None, max_weights_to_save=2):
        self.saved_weights_paths = []
        self.max_weights_to_save = max_weights_to_save
        self.logger = logger
        if logger is None:
            self.logger = configure_logging()

    def __call__(self, save_path):
        self.saved_weights_paths.append(save_path)
        if len(self.saved_weights_paths) > self.max_weights_to_save:
            old_weights_path = self.saved_weights_paths.pop(0)
            if os.path.exists(old_weights_path):
                os.remove(old_weights_path)
                self.logger.info(f"Weigths removed '{old_weights_path}'")


def load_pretrain_model(weights_path, model, logger=None):
    """Load the entire pretrain model or as many layers as possible.
    """
    old_dict = torch.load(weights_path)
    new_dict = model.state_dict()
    if logger is None:
        logger = configure_logging()
    for key, weights in new_dict.items():
        if key in old_dict:
            if new_dict[key].shape == old_dict[key].shape:
                new_dict[key] = old_dict[key]
            else:
                logger.info('Weights {} were not loaded'.format(key))
        else:
            logger.info('Weights {} were not loaded'.format(key))
    return new_dict
