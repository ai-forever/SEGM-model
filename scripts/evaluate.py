import torch
import argparse

from segm.dataset import get_data_loader
from segm.transforms import get_image_transforms, get_mask_transforms
from segm.config import Config
from segm.losses import FbBceLoss
from segm.models import LinkResNet
from segm.utils import val_loop, configure_logging


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_loader(config):
    mask_transforms = get_mask_transforms()
    image_transforms = get_image_transforms()

    test_loader = get_data_loader(
        train_transforms=None,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms,
        csv_paths=config.get_test_datasets('processed_data_path'),
        dataset_probs=config.get_test_datasets('prob'),
        epoch_size=config.get_test('epoch_size'),
        batch_size=config.get_test('batch_size'),
        drop_last=False
    )
    return test_loader


def main(args):
    config = Config(args.config_path)
    test_loader = get_loader(config)
    logger = configure_logging()

    class_names = config.get_classes().keys()
    model = LinkResNet(output_channels=len(class_names))
    model.load_state_dict(torch.load(args.model_path))
    model.to(DEVICE)

    criterion = FbBceLoss()

    val_loop(test_loader, model, criterion, DEVICE, class_names, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/segm_config.json',
                        help='Path to config.json.')
    parser.add_argument('--model_path', type=str,
                        help='Path to model weights.')
    args = parser.parse_args()

    main(args)
