import torch
from torch.utils.data import Dataset

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
        csv_data['file_name'] = csv_data['file_name'].apply(
            get_full_img_path, csv_path=csv_path)
        csv_data['srink_mask_name'] = csv_data['srink_mask_name'].apply(
            get_full_img_path, csv_path=csv_path)
        csv_data['border_mask_name'] = csv_data['border_mask_name'].apply(
            get_full_img_path, csv_path=csv_path)
        data.append(
            csv_data[['file_name', 'dataset_name', 'srink_mask_name',
                      'border_mask_name']]
        )
    data = pd.concat(data, ignore_index=True)
    return data


class SEGMDataset(Dataset):
    """torch.Dataset for segmentation model.

    Args:
        data (pandas.DataFrame): Dataset with 'file_name', 'srink_mask_name'
            and 'border_mask_name' columns with relative paths to images and
            target masks.
        transform (torchvision.Compose): Image transforms, default is None.
    """

    def __init__(self, data, image_transforms=None, mask_transforms=None):
        super().__init__()
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.data_len = len(data)
        self.img_paths = data['file_name'].values
        self.shrink_paths = data['srink_mask_name'].values
        self.border_paths = data['border_mask_name'].values

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        shrink_path = self.shrink_paths[idx]
        border_path = self.border_paths[idx]
        image = cv2.imread(img_path)
        shrink_mask = np.load(shrink_path)
        border_mask = np.load(border_path)
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        if self.mask_transforms is not None:
            shrink_mask = self.mask_transforms(shrink_mask)
            border_mask = self.mask_transforms(border_mask)
        return image, shrink_mask, border_mask


def is_valid_polygon(polygon):
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
        thresh_min (Float): Threshold of the border mask.
        thresh_max (Float): Threshold of the border mask.
    """

    def __init__(
        self, image_h, image_w, shrink_ratio, thresh_min=0.3, thresh_max=0.7
    ):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.canvas = np.zeros((image_h, image_w), dtype=np.float32)
        self.mask = np.zeros((image_h, image_w), dtype=np.float32)

    def get_border_mask(self):
        canvas = self.canvas
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
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
        """Add new polygon border to the mask.

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
        cv2.fillPoly(self.mask, [padded_polygon.astype(np.int32)], 1.0)
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
