import torch
import argparse

from segm.src.dataset import read_and_concat_datasets, SEGMDataset
from segm.src.transforms import get_image_transforms, get_mask_transforms
from segm.src.config import Config
from segm.src.losses import FbBceLoss
from segm.src.models import LinkResNet
from segm.src.utils import val_loop


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_loader(config):
    mask_transforms = get_mask_transforms()
    image_transforms = get_image_transforms()

    data = read_and_concat_datasets([config.get_test('processed_data_path')])
    test_dataset = SEGMDataset(
        data=data,
        train_transforms=None,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.get_test('batch_size'),
        num_workers=5,
    )
    return test_loader


def main(args):
    config = Config(args.config_path)
    test_loader = get_loader(config)

    model = LinkResNet()
    model.load_state_dict(torch.load(args.model_path))
    model.to(DEVICE)

    criterion = FbBceLoss()

    val_loop(test_loader, model, criterion, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/segm/config.json',
                        help='Path to config.json.')
    parser.add_argument('--model_path', type=str,
                        help='Path to model weights.')
    args = parser.parse_args()

    main(args)
