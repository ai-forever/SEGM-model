import torch
import time
from tqdm import tqdm

from utils.utils import AverageMeter, sec2min
from segm.src.metrics import get_iou, get_f1_score


def val_loop(data_loader, model, criterion, device, threshold=0.5):
    loss_avg = AverageMeter()
    iou_avg = AverageMeter()
    f1_score_avg = AverageMeter()
    strat_time = time.time()
    model.eval()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    with torch.no_grad():
        for images, targets in tqdm_data_loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = len(images)
            preds = model(images)

            loss = criterion(preds, targets)
            loss_avg.update(loss.item(), batch_size)

            iou = get_iou(preds, targets, threshold)
            iou_avg.update(iou, batch_size)
            f1_score = get_f1_score(preds, targets, threshold)
            f1_score_avg.update(f1_score, batch_size)
    loop_time = sec2min(time.time() - strat_time)
    print(f'Validation, '
          f'Loss: {loss_avg.avg:.4f}, '
          f'IOU threshold {threshold}: {iou_avg.avg:.4f}, '
          f'F1 score: {f1_score_avg.avg:.4f}, '
          f'loop_time: {loop_time}')
    return loss_avg.avg
