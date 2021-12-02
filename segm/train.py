from tqdm import tqdm
import os
import time
import numpy as np
import torch
import argparse

from utils.utils import (
    load_pretrain_model, FilesLimitControl, AverageMeter, sec2min
)
from segm.src.dataset import read_and_concat_datasets, SEGMDataset
from segm.src.transforms import (
    get_train_transforms, get_val_transforms, get_mask_transforms
)
from segm.src.config import Config
from segm.src.models import DBnet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(
    data_loader, model, shrink_criterion, border_criterion,
    binary_criterion, optimizer, epoch
):
    loss_avg = AverageMeter()
    strat_time = time.time()
    model.train()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, shrink_targets, border_targets in tqdm_data_loader:
        model.zero_grad()
        images = images.to(DEVICE)
        shrink_targets = shrink_targets.to(DEVICE)
        border_targets = border_targets.to(DEVICE)
        batch_size = len(images)
        shrink_pred, border_pred, binary_pred = model(images)

        shrink_loss = shrink_criterion(shrink_pred, shrink_targets)
        border_loss = border_criterion(border_pred, border_targets)
        binary_loss = binary_criterion(binary_pred, shrink_targets)
        loss = shrink_loss + binary_loss + 10 * border_loss
        loss_avg.update(loss.item(), batch_size)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
    loop_time = sec2min(time.time() - strat_time)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print(f'\nEpoch {epoch}, Loss: {loss_avg.avg:.5f}, '
          f'LR: {lr:.7f}, loop_time: {loop_time}')


def val_loop(
    data_loader, model, shrink_criterion, border_criterion, binary_criterion
):
    loss_avg = AverageMeter()
    strat_time = time.time()
    model.eval()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    with torch.no_grad():
        for images, shrink_targets, border_targets in tqdm_data_loader:
            images = images.to(DEVICE)
            shrink_targets = shrink_targets.to(DEVICE)
            border_targets = border_targets.to(DEVICE)
            batch_size = len(images)
            shrink_pred, border_pred, binary_pred = model(images)

            shrink_loss = shrink_criterion(shrink_pred, shrink_targets)
            border_loss = border_criterion(border_pred, border_targets)
            binary_loss = binary_criterion(binary_pred, shrink_targets)
            loss = shrink_loss + binary_loss + 10 * border_loss
            loss_avg.update(loss.item(), batch_size)
    loop_time = sec2min(time.time() - strat_time)
    print(f'Validation, '
          f'Loss: {loss_avg.avg:.4f}, '
          f'loop_time: {loop_time}')
    return loss_avg.avg


def get_loaders(config):
    mask_transforms = get_mask_transforms()

    data = read_and_concat_datasets([config.get_train('processed_data_path')])
    train_transforms = get_train_transforms()
    train_dataset = SEGMDataset(data, train_transforms, mask_transforms)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.get_train('batch_size'),
        num_workers=8,
    )

    data = read_and_concat_datasets([config.get_val('processed_data_path')])
    val_transforms = get_val_transforms()
    val_dataset = SEGMDataset(data, val_transforms, mask_transforms)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.get_val('batch_size'),
        num_workers=8,
    )
    return train_loader, val_loader


def main(args):
    config = Config(args.config_path)
    os.makedirs(config.get('save_dir'), exist_ok=True)

    train_loader, val_loader = get_loaders(config)

    model = DBnet()
    if config.get('pretrain_path'):
        states = load_pretrain_model(config.get('pretrain_path'), model)
        model.load_state_dict(states)
        print('Load pretrained model')
    model.to(DEVICE)

    shrink_criterion = torch.nn.BCELoss()
    border_criterion = torch.nn.L1Loss()
    binary_criterion = torch.nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=15)

    weight_limit_control = FilesLimitControl()
    best_loss = np.inf
    val_loss = val_loop(val_loader, model, shrink_criterion, border_criterion,
                        binary_criterion)
    for epoch in range(config.get('num_epochs')):
        train_loop(train_loader, model, shrink_criterion, border_criterion,
                   binary_criterion, optimizer, epoch)
        val_loss = val_loop(val_loader, model, shrink_criterion,
                            border_criterion, binary_criterion)
        scheduler.step(val_loss)
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
                        default='/workdir/segm/config.json',
                        help='Path to config.json.')
    args = parser.parse_args()

    main(args)
