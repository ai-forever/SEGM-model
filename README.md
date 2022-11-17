# Segmentation model

This is a model for semantic segmentation based on [LinkNet](https://arxiv.org/abs/1707.03718) (Unet-like architecture).

SEGM-model is a part of [ReadingPipeline](https://github.com/ai-forever/ReadingPipeline) repo.

## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

### Preparations

- Clone the repo.
- Download and extract dataset to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
  Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

If you don't want to use Docker, you can install dependencies via requirements.txt

## Configuring the model

You can change the [segm_config.json](scripts/segm_config.json) and set some of the the base training and evaluating parameters: num epochs, image size, saving dir, etc.

### Class specific parameters

Parameters in the `classes` are set individually for each class of the model. The order of the subdicts in the `classes` corresponds to the order of the mask layers in the predicted tensor. Each dictionary contains parameters for model classes to pre- and post-process stages, for example:

```
"classes": {
	"pupil_and_teacher_comments": {
		"annotation_classes": ["pupil_comment", "teacher_comment"],
		"polygon2mask": {
			"ShrinkMaskMaker": {"shrink_ratio": 0.5}
		},
		"postprocess": {
			"threshold": 0.8,
			"min_area": 10
		}
	},
	...
}
```

- `annotation_classes` - a list with class names from `annotation["categories"]`. Polygons of these classes will be combined into one class.
- `polygon2mask` - a list of functions that will be applied one by one to convert polygons to mask and prepare target for this class. There are several functions available - to create regular or shrinked masks. To add a new function to the processing, you need to add it to the `PREPROCESS_FUNC` dictionary in [prepare_dataset.py](scripts/prepare_dataset.py) and also specify it in the `polygon2mask`-dict in the config.

Postprocessing settings:

- `threshold` is the threshold of the model's confidence. Above this value the mask becomes Ture, below - False. It helps to remove some false predictions of the model with low confidence.
- `min_area` - the minimum area of a polygon (polygons with less area will be removed).

### Dataset folders

Individual for train / val / test:

```
"train": {
    "datasets": [
        {
            "json_path": "path/to/annotaion.json",
            "image_root": "path/to/folder/with/images",
            "processed_data_path": "path/to/save/processed_dataset.csv"
        },
        ...
    ],
    "batch_size": 8
}
```
In `datasets`-dict, you can specify paths to multiple datasets for training and testing.

- `json_path` (to the annotation.json) and `image_root` (to the folder with images) are paths to the dataset with markup in COCO format.
- `processed_data_path` - the saving path of the final csv file, which is produced by the prepare_dataset.py script. This csv-file will be used in the train stage. This file stores paths to the processed target masks.

## Input dataset description

The input dataset should be in COCO format. The `annotation.json` should have the following dictionaries:

- `annotation["categories"]` - a list of dicts with a categories info (categotiy names and indexes).
- `annotation["images"]` - a list of dictionaries with a description of images, each dictionary must contain fields:
  - `file_name` - name of the image file.
  - `id` for image id.
- `annotation["annotations"]` - a list of dictioraties with a murkup information. Each dictionary stores a description for one polygon from the dataset, and must contain the following fields:
  - `image_id` - the index of the image on which the polygon is located.
  - `category_id` - the polygon’s category index.
  - `segmentation` - the coordinates of the polygon, a list of numbers - which are coordinate pairs x and y.

## Prepare target masks

To preprocess dataset and create target masks for training:

```bash
python scripts/prepare_dataset.py --config_path path/to/the/segm_config.json
```

The script creates a target masks for train/val/test stages. The path to the input dataset is set in the config file in `json_path` and `image_root`. The output csv file is saved to `processed_data_path` from the config.

## Training

To train the model:

```bash
python scripts/train.py --config_path path/to/the/segm_config.json
```

## Evaluating

To test the model:

```bash
python scripts/evaluate.py \
--config_path path/to/the/segm_config.json \
--model_path path/to/the/model-weights.ckpt
```

## ONNX

You can convert Torch model to ONNX to speed up inference on cpu.

```bash
python scripts/torch2onnx.py \
--config_path path/to/the/ocr_config.json \
--model_path path/to/the/model-weights.ckpt
```
