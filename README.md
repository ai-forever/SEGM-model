# Segmentation Unet model

## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

Also you can install the necessary python packages via [requirements.txt](requirements.txt)

### Preparations

- Clone the repo.
- Download and extract dataset to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
  Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

## Configuring the model

You can change the [segm_config.json](scripts/segm_config.json) (or make a copy of the file) and set some of the the base training and evaluating parameters: num epochs, image size, saving dir and etc.

### Class specific parameters

Parameters in the "classes"-dict are set individually for each class of the model. The order of the sub-dicts in the "classes"-dict corresponds to the order of the mask layers in the predicted tensor. Each dictionary contains parameters for model classes to pre- and post-process stages, for example:

```
"classes": {
	"pupil_and_teacher_comments": {
		"annotation_classes": ["pupil_comment", "teacher_comment"],
		"polygon2mask": {
			"ShrinkMaskMaker": {"shrink_ratio": 0.5}
		},
		"postprocess": {
			"threshold": 0.8,
			"min_area": 10,
			"upscale_bbox": [1.2, 1.2]
		}
	},
	...
}
```

- `annotation_classes` - a list with class names from `annotation["categories"]` (’name’ key) indicating which polygons from annotation.json will be converted to a target mask. Polygons with these `class names` will be combined into one class mask.
- `polygon2mask` - a list of function that would be applied one by one to convert polygons to mask and prepare target for this class. There are several functions available - to create shrink or border masks. All these functions should be listed in PREPROCESS_FUNC in [prepare_dataset.py](scripts/prepare_dataset.py).

Prediction postprocessing settings:

- `threshold` is the threshold of the model's confidence, above this value the mask becomes Ture, below - False. It helps to remove some false predictions of the model with low confidence.
- `min_area` - the minimum area of the polygon so that it is considered as real, true positive polygon.
- `upscale_bbox` - Tuple of (x, y) upscale parameters of the predicted bbox to increase it and capture large areas of the image.

### Dataset folders

Individual for train / val / test:

```
"train": {
	"json_path": "path/to/annotaion.json",
	"image_root": "path/to/folder/with/images",
	"processed_data_path": "path/to/save/processed/dataset.csv",
	"batch_size": 8
}
```
- `json_path` (to the annotation.json) and `image_root` (to the folder with images) are paths to the dataset with markup in COCO format.
- `processed_data_path` - the saving path of the final csv file, to be produced by the prepare_dataset.py script. This csv-file will be used in train stage. This file store paths to the processed target masks.

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

The script creates a target masks for train, val and test stage. The path to the input dataset is set in the config file in `json_path` and `image_root`. The output csv file is saved to `processed_data_path` from the config.

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
