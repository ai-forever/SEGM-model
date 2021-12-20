import torch

import cv2
import numpy as np

from segm.transforms import InferenceTransform
from segm.models import LinkResNet
from segm.config import Config


def get_bbox_and_contours_from_mask(pred, config, image_height, image_width):
    contours = get_contours_from_mask(pred, config.get('min_area'))
    contours = rescale_contours(
        contours, config.get_image('height'), config.get_image('width'),
        image_height, image_width
    )
    bboxs = get_bbox_from_contours(contours)
    return bboxs, contours


def mask_preprocess(pred, threshold):
    """Mask thresholding and move to cpu and numpy."""
    pred = pred > threshold
    pred = pred.cpu().numpy()
    return pred


class SegmPredictor:
    def __init__(self, model_path, config_path, device='cuda'):
        self.config = Config(config_path)
        self.device = torch.device(device)
        # load model
        self.model = LinkResNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=self.config.get_image('height'),
            width=self.config.get_image('width'),
        )

    def __call__(self, images):
        """Make segmentation prediction.

        Args:
            images (np.ndarray or list of np.ndarray): One image or list of
                images.
        Returns:
            pred_data (dict or list of dicts): A result dict for one input
                image, and a list with dicts if there is a list of input images.
            [
                {
                    'image': {'height': Int, 'width': Int},
                    'predictions': [
                        {
                            'bbox': bbox,
                            'contour': contour
                        },
                        ...
                    ]

                },
                ...
            ]
        """
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        pred_data = []
        for image in images:
            h, w = image.shape[:2]
            pred_data.append(
                {
                    'image': {'height': h, 'width': w},
                    'predictions': []
                }
            )

        images = self.transforms(images)
        images = images.to(self.device)
        preds = self.model(images)
        preds = mask_preprocess(preds, self.config.get('threshold'))

        for image_idx, pred in enumerate(preds):  # iterate through images
            pred = pred[0]  # get zero channel mask
            bboxs, contours = get_bbox_and_contours_from_mask(
                pred=pred,
                config=self.config,
                image_height=pred_data[image_idx]['image']['height'],
                image_width=pred_data[image_idx]['image']['width']
            )
            for bbox, contour in zip(bboxs, contours):
                upscaled_bbox = upscale_bbox(
                    bbox=bbox,
                    upscale_x=self.config.get('upscale_bbox')[0],
                    upscale_y=self.config.get('upscale_bbox')[1]
                )
                pred_data[image_idx]['predictions'].append(
                    {
                        'bbox': upscaled_bbox,
                        'contour': contour
                    }
                )

        if one_image:
            return pred_data[0]
        else:
            return pred_data


def contour2bbox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_bbox_from_contours(contours):
    bboxs = []
    if len(contours) > 0:
        for contour in contours:
            bboxs.append(contour2bbox(contour))
    return bboxs


def rescale_contours(
    contours, pred_height, pred_width, image_height, image_width
):
    """Rescale contours from prediction mask shape to input image size."""
    y_ratio = image_height / pred_width
    x_ratio = image_width / pred_height
    scale = (x_ratio, y_ratio)
    for contour in contours:
        for i in range(2):
            contour[:, :, i] = contour[:, :, i] * scale[i]
    return contours


def rescale_bbox(bboxes, pred_height, pred_width, image_height, image_width):
    """Rescale bbox from prediction mask shape to input image size."""
    y_ratio = image_height / pred_width
    x_ratio = image_width / pred_height
    scale = (x_ratio, y_ratio)
    rescaled_bboxes = []
    for bbox in bboxes:
        rescaled_bbox = [int(round(bbox[i] * scale[i % 2]))
                         for i in range(4)]
        rescaled_bboxes.append(tuple(rescaled_bbox))
    return rescaled_bboxes


def upscale_bbox(bbox, upscale_x=1, upscale_y=1):
    """Increase size of the bbox."""
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]

    y_change = (height * upscale_y) - height
    x_change = (width * upscale_x) - width

    x_min = max(0, bbox[0] - int(x_change/2))
    y_min = max(0, bbox[1] - int(y_change/2))
    x_max = bbox[2] + int(x_change/2)
    y_max = bbox[3] + int(y_change/2)
    return x_min, y_min, x_max, y_max
