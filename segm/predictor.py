import torch
import onnxruntime as ort
import openvino.runtime as ov
from enum import Enum

import cv2
import numpy as np

from segm.transforms import InferenceTransform
from segm.models import LinkResNet
from segm.config import Config


def predict(images, model, device, targets=None):
    """Make model prediction.
    Args:
        images (torch.Tensor): Batch with tensor images.
        model (ocr.src.models.CRNN): OCR model.
        device (torch.device): Torch device.
        targets (torch.Tensor): Batch with tensor masks. By default is None.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)

    if targets is not None:
        targets = targets.to(device)
        return output, targets
    return output


def contour2bbox(contour):
    """Get bbox from contour."""
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


class SegmModel:
    def predict(self):
        raise NotImplementedError

    def get_preds(self):
        raise NotImplementedError


def get_preds(images, preds, cls2params, config, cuda_torch_input=True):
    pred_data = []
    for image, pred in zip(images, preds):  # iterate through images
        img_h, img_w = image.shape[:2]
        pred_img = {'predictions': []}
        for cls_idx, cls_name in enumerate(cls2params):  # iter through classes
            pred_cls = pred[cls_idx]
            # thresholding works faster on cuda than on cpu
            pred_cls = \
                pred_cls > cls2params[cls_name]['postprocess']['threshold']
            if cuda_torch_input:
                pred_cls = pred_cls.cpu().numpy()

            contours = get_contours_from_mask(
                pred_cls, cls2params[cls_name]['postprocess']['min_area'])
            contours = rescale_contours(
                contours=contours,
                pred_height=config.get_image('height'),
                pred_width=config.get_image('width'),
                image_height=img_h,
                image_width=img_w
            )
            bboxes = [contour2bbox(contour) for contour in contours]
            contours = reduce_contours_dims(contours)

            for contour, bbox in zip(contours, bboxes):
                pred_img['predictions'].append(
                    {
                        'polygon': contour,
                        'bbox': bbox,
                        'class_name': cls_name
                    }
                )
        pred_data.append(pred_img)
    return pred_data


class SegmONNXCPUModel(SegmModel):
    def __init__(self, model_path, num_threads, config_path):
        self.config = Config(config_path)
        self.cls2params = self.config.get_classes()
        sess = ort.SessionOptions()
        sess.intra_op_num_threads = num_threads
        sess.inter_op_num_threads = num_threads
        self.model = ort.InferenceSession(model_path, sess)
        self.transforms = InferenceTransform(
            height=self.config.get_image('height'),
            width=self.config.get_image('width'),
            return_numpy=True
        )

    def predict(self, images):
        transformed_images = self.transforms(images)
        output = self.model.run(
            None,
            {"input": transformed_images},
        )[0]
        return output

    def get_preds(self, images, preds):
        pred_data = get_preds(
            images, preds, self.cls2params, self.config, False)
        return pred_data


class SegmOpenVINOCPUModel(SegmModel):
    def __init__(self, model_path, num_threads, config_path):
        self.config = Config(config_path)
        self.cls2params = self.config.get_classes()
        ie = ov.Core()
        model_onnx = ie.read_model(model_path, "AUTO")
        self.model = ie.compile_model(
            model=model_onnx,
            device_name="CPU",
            config={"INFERENCE_NUM_THREADS": str(num_threads)}
        )
        self.transforms = InferenceTransform(
            height=self.config.get_image('height'),
            width=self.config.get_image('width'),
            return_numpy=True
        )

    def predict(self, images):
        transformed_images = self.transforms(images)
        infer_request = self.model.create_infer_request()
        infer_request.infer([transformed_images])
        output = infer_request.get_output_tensor().data
        return output

    def get_preds(self, images, preds):
        pred_data = get_preds(
            images, preds, self.cls2params, self.config, False)
        return pred_data


class SegmTorchModel(SegmModel):
    def __init__(self, model_path, config_path, device='cuda'):
        self.config = Config(config_path)
        self.device = torch.device(device)
        self.cls2params = self.config.get_classes()
        # load model
        self.model = LinkResNet(
            output_channels=len(self.cls2params),
            pretrained=False
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=self.config.get_image('height'),
            width=self.config.get_image('width'),
        )

    def predict(self, images):
        transformed_images = self.transforms(images)
        preds = predict(transformed_images, self.model, self.device)
        return preds

    def get_preds(self, images, preds):
        pred_data = get_preds(images, preds, self.cls2params, self.config)
        return pred_data


class RuntimeType(Enum):
    ONNX = "ONNX"
    OVINO = "OpenVino"
    TORCH = "Pytorch"


def validate_value_in_enum(value, enum_cls: Enum):
    enum_values = [e.value for e in enum_cls]
    if value not in enum_values:
        raise Exception(f"{value} is not supported. "
                        f"Allowed types are: {', '.join(enum_values)}")


class SegmPredictor:
    """Make SEGM prediction.

    Args:
        model_path (str): The path to the model weights.
        config_path (str): The path to the model config.
        num_threads (int): The number of cpu threads to use
            (in ONNX and OpenVino runtimes).
        runtime (str): The runtime method of the model (Pytorch, ONNX or
            OpenVino from the RuntimeType). Default is Pytorch.
        device (str): The device for computation. Default is cuda.
    """

    def __init__(
        self, model_path, config_path, num_threads, device='cuda',
        runtime='Pytorch'
    ):
        validate_value_in_enum(runtime, RuntimeType)
        if RuntimeType(runtime) is RuntimeType.TORCH:
            self.model = SegmTorchModel(model_path, config_path, device)
        elif (
            RuntimeType(runtime) is RuntimeType.ONNX
            and device == 'cpu'
        ):
            self.model = SegmONNXCPUModel(model_path, num_threads, config_path)
        elif (
            RuntimeType(runtime) is RuntimeType.OVINO
            and device == 'cpu'
        ):
            self.model = SegmOpenVINOCPUModel(
                model_path, num_threads, config_path)
        else:
            raise Exception(f"Runtime type {runtime} with device {device} "
                            "are not supported options.")

    def __call__(self, images):
        """
        Args:
            images (list of np.ndarray): A list of images in BGR format.

        Returns:
            pred_data (dict or list of dicts): A result dict for one input
                image, and a list with dicts if there was a list with images.
            [
                {
                    'predictions': [
                        {
                            'polygon': polygon [[x1,y1], [x2,y2], ..., [xN,yN]]
                            'bbox': bounding box [x_min, y_min, x_max, y_max]
                            'class_name': str, class name of the polygon.
                        },
                        ...
                    ]

                },
                ...
            ]
        """
        preds = self.model.predict(images)
        pred_data = self.model.get_preds(images, preds)
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
