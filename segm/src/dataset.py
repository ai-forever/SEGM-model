import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon


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
        self.shrink_mask = np.zeros(shape=(image_h, image_w), dtype=np.float32)

    def add_polygon_to_mask(self, polygon):
        area = Polygon(polygon).area
        peri = cv2.arcLength(polygon, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(polygon, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
        distance = int(area * (1 - self.shrink_ratio ** 2) / peri)
        shrinked_bbox = pco.Execute(-distance)
        shrinked_bbox = np.array(shrinked_bbox)[0]
        cv2.fillPoly(self.shrink_mask, [shrinked_bbox], (1.0))

    def get_shrink_mask(self):
        return self.shrink_mask
