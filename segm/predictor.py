import torch

import cv2
import numpy as np

from segm.transforms import InferenceTransform
from segm.models import LinkResNet
from segm.config import Config


def mask_preprocess(pred, threshold):
    """Mask thresholding and move to cpu and numpy."""
    pred = pred > threshold
    pred = pred.cpu().numpy()
    return pred


def get_contours_from_predictions(
    pred, class_params, pred_height, pred_width, image_height, image_width
):
    """Process predictions and return contours."""
    pred = mask_preprocess(pred, class_params['postprocess']['threshold'])
    contours = get_contours_from_mask(
        mask=pred,
        min_area=class_params['postprocess']['min_area']
    )
    contours = rescale_contours(
        contours=contours,
        pred_height=pred_height,
        pred_width=pred_width,
        image_height=image_height,
        image_width=image_width
    )
    contours = reduce_contours_dims(contours)
    return contours


class SegmPredictor:
    def __init__(self, model_path, config_path, device='cuda'):
        self.config = Config(config_path)
        self.device = torch.device(device)
        self.cls2params = self.config.get_classes()
        # load model
        self.model = LinkResNet(output_channels=len(self.cls2params))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

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
                            'polygon': polygon [ [x1,y1], [x2,y2], ..., [xN,yN] ]
                            'class_name': str, class name of the polygon.
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

        transformed_images = self.transforms(images)
        transformed_images = transformed_images.to(self.device)
        with torch.no_grad():
            preds = self.model(transformed_images)

        pred_data = []
        for image, pred in zip(images, preds):  # iterate through images
            img_h, img_w = image.shape[:2]
            pred_img = {
                'image': {'height': img_h, 'width': img_w},
                'predictions': []
            }
            for cls_idx, cls_name in enumerate(self.cls2params):  # iterate through classes
                # prediction processing
                contours = get_contours_from_predictions(
                    pred=pred[cls_idx],
                    class_params=self.cls2params[cls_name],
                    pred_height=self.config.get_image('height'),
                    pred_width=self.config.get_image('width'),
                    image_height=img_h,
                    image_width=img_w
                )
                # put predictions in output json
                for contour in contours:
                    pred_img['predictions'].append(
                        {
                            'polygon': contour,
                            'class_name': cls_name
                        }
                    )
        pred_data.append(pred_img)

        if one_image:
            return pred_data[0]
        else:
            return pred_data


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def rescale_contours(
    contours, pred_height, pred_width, image_height, image_width
):
    """Rescale contours from prediction mask shape to input image size."""
    y_ratio = image_height / pred_height
    x_ratio = image_width / pred_width
    scale = (x_ratio, y_ratio)
    for contour in contours:
        for i in range(2):
            contour[:, :, i] = contour[:, :, i] * scale[i]
    return contours


def reduce_contours_dims(contours):
    reduced_contours = []
    for contour in contours:
        contour = [[int(i[0][0]), int(i[0][1])] for i in contour]
        reduced_contours.append(contour)
    return reduced_contours
