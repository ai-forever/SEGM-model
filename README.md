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

You can change the [segm_config.json](scripts/segm_config.json) (or make a copy of the file) and set the necessary training and evaluating parameters: num epochs, image size, saving path and etc.

Other parameters from config:

- `mask_shrink_ratio` and `border_shrink_ratio` are configuration parameters for creating targets of DBNet-architecture - shrink and border masks.
- `threshold` is the value of the model's confidence, above this value the mask becomes Ture, below - False.
- `min_area` - the minimum area of the polygon so that it is considered as real, true positive polygon.

Dataset parameters:

```
"train": {
    "json_path": "path/to/annotaion.json",
    "image_root": "path/to/folder/with/images",
    "category_ids": list of category indexes to be trained on,
    "processed_data_path": "path/to/save/processed/dataset.csv"
}
```
- `json_path` (to the annotation.json) and `image_root` (to the folder with images) are paths to the dataset with markup in COCO format.
- `category_ids` - should be a list with category indexes indicating which polygons from annotation will be converted to a target mask for training the model. 
- `processed_data_path` - the saving path of the final csv file, to be produced by the prepare_dataset.py script. This csv-file will be used in train stage. This file store paths to the processed target masks.

## Input dataset description

The input dataset should be in COCO format. The `annotation.json` should have the following dictionaries:

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

### Training

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
