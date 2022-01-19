from tqdm import tqdm
import os
import time
import numpy as np
import torch
import argparse

from segm.utils import (
    val_loop, load_pretrain_model, FilesLimitControl, sec2min
)
from segm.dataset import read_and_concat_datasets, SEGMDataset
from segm.transforms import (
    get_train_transforms, get_image_transforms, get_mask_transforms
)
from segm.config import Config
from segm.metrics import get_iou, get_f1_score, AverageMeter, IOUMetric
from segm.losses import FbBceLoss
from segm.models import LinkResNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(data_loader, model, criterion, optimizer, epoch, class_names):
    loss_avg = AverageMeter()
    iou_avg = AverageMeter()
    cls2iou = {cls_name: IOUMetric(cls_idx)
               for cls_idx, cls_name in enumerate(class_names)}
    f1_score_avg = AverageMeter()
    strat_time = time.time()
    model.train()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, targets in tqdm_data_loader:
        model.zero_grad()
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        batch_size = len(images)
        preds = model(images)

        loss = criterion(preds, targets)
        loss_avg.update(loss.item(), batch_size)

        iou_avg.update(get_iou(preds, targets), batch_size)
        f1_score_avg.update(get_f1_score(preds, targets), batch_size)
        for cls_name in class_names:
            cls2iou[cls_name](preds, targets)

        loss.backward()
        optimizer.step()
    loop_time = sec2min(time.time() - strat_time)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    cls2iou_log = ''.join([f' IOU {cls_name}: {iou_fun.avg():.4f}'
                           for cls_name, iou_fun in cls2iou.items()])
    print(f'\nEpoch {epoch}, '
          f'Loss: {loss_avg.avg:.5f}, '
          f'IOU avg: {iou_avg.avg:.4f}, '
          f'{cls2iou_log}, '
          f'F1 avg: {f1_score_avg.avg:.4f}, '
          f'LR: {lr:.7f}, '
          f'loop_time: {loop_time}')
    return loss_avg.avg


def get_loaders(config):
    mask_transforms = get_mask_transforms()
    image_transforms = get_image_transforms()

    data = read_and_concat_datasets([config.get_train('processed_data_path')])
    train_transforms = get_train_transforms(config.get_image('height'),
                                            config.get_image('width'))
    train_dataset = SEGMDataset(
        data=data,
        train_transforms=train_transforms,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.get_train('batch_size'),
        num_workers=5,
        shuffle=True
    )

    data = read_and_concat_datasets([config.get_val('processed_data_path')])
    val_dataset = SEGMDataset(
        data=data,
        train_transforms=None,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.get_val('batch_size'),
        num_workers=5,
    )
    return train_loader, val_loader


def main(args):
    config = Config(args.config_path)
    os.makedirs(config.get('save_dir'), exist_ok=True)

    train_loader, val_loader = get_loaders(config)

    class_names = config.get_classes().keys()
    model = LinkResNet(output_channels=len(class_names))
    if config.get('pretrain_path'):
        states = load_pretrain_model(config.get('pretrain_path'), model)
        model.load_state_dict(states)
        print('Load pretrained model')
    model.to(DEVICE)

    criterion = FbBceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=25)

    weight_limit_control = FilesLimitControl()
    best_loss = np.inf
    val_loss = val_loop(val_loader, model, criterion, DEVICE, class_names)
    for epoch in range(config.get('num_epochs')):
        train_loss = train_loop(train_loader, model, criterion, optimizer,
                                epoch, class_names)
        val_loss = val_loop(val_loader, model, criterion, DEVICE, class_names)
        scheduler.step(train_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            model_save_path = os.path.join(
                config.get('save_dir'), f'model-{epoch}-{val_loss:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')
            weight_limit_control(model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/segm_config.json',
                        help='Path to config.json.')
    args = parser.parse_args()

    main(args)
