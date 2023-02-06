import torch
from torch.utils.data import Dataset, Sampler

import cv2
from pathlib import Path
import pandas as pd
import pyclipper
import numpy as np
from shapely.geometry import Polygon


def get_full_img_path(img_root_path, csv_path):
    """Merge csv root path and image name."""
    root_dir = Path(csv_path).parent
    img_path = root_dir / Path(img_root_path)
    return str(img_path)


def read_and_concat_datasets(csv_paths):
    """Read csv files and concatenate them into one pandas DataFrame.

    Args:
        csv_paths (list): List of the dataset csv paths.

    Return:
        data (pandas.DataFrame): Concatenated datasets.
    """
    data = []
    for csv_path in csv_paths:
        csv_data = pd.read_csv(csv_path)
        csv_data['dataset_name'] = csv_path
        csv_data['file_name'] = csv_data['image'].apply(
            get_full_img_path, csv_path=csv_path)
        csv_data['target'] = csv_data['target'].apply(
            get_full_img_path, csv_path=csv_path)
        data.append(
            csv_data[['file_name', 'dataset_name', 'target']]
        )
    data = pd.concat(data, ignore_index=True)
    return data


def get_data_loader(
    train_transforms, image_transforms, mask_transforms, csv_paths,
    dataset_probs, epoch_size, batch_size, drop_last
):
    data = read_and_concat_datasets(csv_paths)
    dataset_prob2sample_prob = DatasetProb2SampleProb(csv_paths, dataset_probs)
    data = dataset_prob2sample_prob(data)

    dataset = SEGMDataset(
        data=data,
        train_transforms=train_transforms,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms
    )
    sampler = SequentialSampler(
        dataset_len=len(data),
        epoch_size=epoch_size,
        init_sample_probs=data['sample_prob'].values
    )
    batcher = torch.utils.data.BatchSampler(sampler, batch_size=batch_size,
                                            drop_last=drop_last)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batcher,
        num_workers=8,
    )
    return data_loader


class DatasetProb2SampleProb:
    """Convert dataset sampling probability to probability for each sample
    in the datset.

    Args:
        dataset_names (list): A list of the dataset names.
        dataset_probs (list of float): A list of dataset sample probs
            corresponding to the datasets from dataset_names list.
    """

    def __init__(self, dataset_names, dataset_probs):
        assert len(dataset_names) == len(dataset_probs), "Length of " \
            "csv_paths should be equal to the length of the dataset_probs."
        self.dataset2dataset_prob = dict(zip(dataset_names, dataset_probs))

    def _dataset2sample_count(self, data):
        """Calculate samples in each dataset from data using."""
        dataset2sample_count = {}
        for dataset_name in self.dataset2dataset_prob:
            dataset2sample_count[dataset_name] = \
                (data['dataset_name'] == dataset_name).sum()
        return dataset2sample_count

    def _dataset2sample_prob(self, dataset2sample_count):
        """Convert dataaset prob to sample prob."""
        dataset2sample_prob = {}
        for dataset_name, dataset_prob in self.dataset2dataset_prob.items():
            sample_count = dataset2sample_count[dataset_name]
            dataset2sample_prob[dataset_name] = dataset_prob / sample_count
        return dataset2sample_prob

    def __call__(self, data):
        """Add sampling prob column to data.

        Args:
            data (pandas.DataFrame): Dataset with 'dataset_name' column.
        """
        dataset2sample_count = self._dataset2sample_count(data)
        dataset2sample_prob = \
            self._dataset2sample_prob(dataset2sample_count)
        data['sample_prob'] = data['dataset_name'].apply(
            lambda x: dataset2sample_prob[x])
        return data


class SequentialSampler(Sampler):
    """Make sequence of dataset indexes for batch sampler.
    Args:
        dataset_len (int): Length of train dataset.
        epoch_size (int, optional): Size of train epoch (by default it
            is equal to the dataset_len). Can be specified if you need to
            reduce the time of the epoch.
        init_sample_probs (list, optional): List of samples' probabilities to
            be added in batch. If None probs for all samples would be the same.
            The length of the list must be equal to the length of the dataset.
    """
    def __init__(self, dataset_len, epoch_size=None, init_sample_probs=None):
        self.dataset_len = dataset_len
        if epoch_size is not None:
            self.epoch_size = epoch_size
        else:
            self.epoch_size = dataset_len

        if init_sample_probs is None:
            self.init_sample_probs = \
                np.array([1. for i in range(dataset_len)], dtype=np.float64)
        else:
            self.init_sample_probs = \
                np.array(init_sample_probs, dtype=np.float64)
            assert len(self.init_sample_probs) == dataset_len, "The len " \
                "of the sample_probs must be equal to the dataset_len."
        self.init_sample_probs = \
            self._sample_probs_normalization(self.init_sample_probs)

    def _sample_probs_normalization(self, sample_probs):
        """Probabilities normalization to make them sum to 1.
        Sum might not be equal to 1 if probs are too small.
        """
        return sample_probs / sample_probs.sum()

    def __iter__(self):
        dataset_indexes = np.random.choice(
            a=self.dataset_len,
            size=self.epoch_size,
            p=self.init_sample_probs,
            replace=False,  # only unique samples inside an epoch
        )
        return iter(dataset_indexes)

    def __len__(self):
        return self.epoch_size


class SEGMDataset(Dataset):
    """torch.Dataset for segmentation model.

    Args:
        data (pandas.DataFrame): Dataset with 'file_name', 'target'
             columns with paths to images and target masks.
        train_transforms (torchvision.Compose): Train images and masks
            transforms, default is None.
        image_transforms (torchvision.Compose): Images transforms,
            default is None.
        mask_transforms (torchvision.Compose): Masks transforms,
            default is None.
    """

    def __init__(
        self, data, train_transforms=None, image_transforms=None,
        mask_transforms=None
    ):
        super().__init__()
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.train_transforms = train_transforms
        self.dataset_len = len(data)
        self.img_paths = data['file_name'].values
        self.target_paths = data['target'].values

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        target_path = self.target_paths[idx]
        image = cv2.imread(img_path)
        target = np.load(target_path)

        if self.train_transforms is not None:
            image, target = self.train_transforms(image, target)
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        if self.mask_transforms is not None:
            target = self.mask_transforms(target)
        return image, target


def is_valid_polygon(polygon):
    """Check if a polygon is valid. Return True if valid and False otherwise.

    Args:
        polygon (shapely.geometry.Polygon): The polygon.
    """
    if (
        polygon.length < 1
        or polygon.area <= 0
    ):
        return False
    return True


class MakeShrinkMask:
    """Shrink polygon and make mask from it.

    This is adapted from:
    https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/textdet_targets/base_textdet_targets.py#L88
    https://github.com/yts2020/DBnet_pytorch/blob/master/DBnet_pytorch.py#L27

    Args:
        image_h (Int): Image height.
        image_w (Int): Image width.
        shrink_ratio (Float): The ratio to shrink the polygon.
    """

    def __init__(self, image_h, image_w, shrink_ratio=0.5):
        self.shrink_ratio = shrink_ratio
        self.shrink_mask = np.zeros(shape=(image_h, image_w), dtype=np.uint8)

    def add_polygon_to_mask(self, polygon):
        """Add new polygon to the mask.

        Args:
            polygon (np.array): Array of polygon coordinates
                np.array([[x, y], ...])
        """
        poly = Polygon(polygon)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(polygon, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
        if not is_valid_polygon(poly):
            return False
        distance = int(poly.area * (1 - self.shrink_ratio ** 2) / poly.length)
        # polygon could be splitted to several parts after shrink operation
        # https://stackoverflow.com/a/33902816
        shrinked_bboxes = pco.Execute(-distance)
        for shrinked_bbox in shrinked_bboxes:
            shrinked_bbox = np.array(shrinked_bbox)
            cv2.fillPoly(self.shrink_mask, [shrinked_bbox], 1)

    def get_shrink_mask(self):
        return self.shrink_mask


class MakeBorderMask:
    """Create a polygon border and put it on the mask.

    This is adapted from:
    https://github.com/yts2020/DBnet_pytorch/blob/master/DBnet_pytorch.py#L66

    Args:
        image_h (Int): Image height.
        image_w (Int): Image width.
        shrink_ratio (Float): The ratio to shrink the border.
    """

    def __init__(self, image_h, image_w, shrink_ratio):
        self.shrink_ratio = shrink_ratio
        self.canvas = np.zeros((image_h, image_w), dtype=np.float32)

    def get_border_mask(self):
        canvas = self.canvas
        canvas = (canvas > 0.5).astype(np.uint8)  # binarize mask
        return canvas

    def _distance_matrix(self, xs, ys, a, b):
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]
        u1 = (((xs - x1) * (x2 - x1)) + ((ys - y1) * (y2 - y1)))
        u = u1 / (np.square(x1 - x2) + np.square(y1 - y2))
        u[u <= 0] = 2
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        distance = np.sqrt(np.square(xs - ix) + np.square(ys - iy))
        distance2 = np.sqrt(np.fmin(np.square(xs - x1) + np.square(ys - y1),
                            np.square(xs - x2) + np.square(ys - y2)))
        distance[u >= 1] = distance2[u >= 1]
        return distance

    def add_border_to_mask(self, polygon):
        """Add new polygon border to the canvas.

        Args:
            polygon (np.array): Array of polygon coordinates
                np.array([[x, y], ...])
        """
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        poly = Polygon(polygon)
        if not is_valid_polygon(poly):
            return False
        distance = poly.area * (1 - np.power(self.shrink_ratio, 2)) / poly.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width))
        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._distance_matrix(
                xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)
        xmin_valid = min(max(0, xmin), self.canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), self.canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), self.canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), self.canvas.shape[0] - 1)
        self.canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            self.canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
