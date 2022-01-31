import numpy as np
import json
import cv2
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import argparse

from segm.config import Config
from segm.dataset import MakeShrinkMask, MakeBorderMask


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [[list_of_numbers[i], list_of_numbers[i+1]]
            for i in range(0, len(list_of_numbers), 2)]
    return np.array(bbox)


def get_shrink_mask(polygons, image_h, image_w, shrink_ratio):
    """To create shrinked masks target."""
    shrink_mask_maker = MakeShrinkMask(image_h, image_w, shrink_ratio)
    for polygon in polygons:
        shrink_mask_maker.add_polygon_to_mask(polygon)
    return shrink_mask_maker.get_shrink_mask()


def get_border_mask(polygons, image_h, image_w, shrink_ratio):
    """To create border masks target."""
    border_mask_maker = MakeBorderMask(image_h, image_w, shrink_ratio)
    for polygon in polygons:
        border_mask_maker.add_border_to_mask(polygon)
    return border_mask_maker.get_border_mask()


def polygon_resize(polygons, old_img_h, old_img_w, new_img_h, new_img_w):
    h_ratio = new_img_h / old_img_h
    w_ratio = new_img_w / old_img_w
    resized_polygons = []
    for polygon in polygons:
        r_p = [(int(x * w_ratio), int(y * h_ratio)) for x, y in polygon]
        resized_polygons.append(np.array(r_p))
    return resized_polygons


def get_class_polygons(image_id, data, image, category_ids):
    """Get polygons from annotation for a specific image and category.
    """
    polygons = []
    for data_ann in data['annotations']:
        if (
            data_ann['image_id'] == image_id
            and data_ann['category_id'] in category_ids
            and data_ann['segmentation']
        ):
            polygon = numbers2coords(data_ann['segmentation'][0])
            polygons.append(polygon)
    return polygons


def class_names2id(class_names, data):
    """Match class names to categoty ids using annotation in COCO format."""
    category_ids = []
    for class_name in class_names:
        for category_info in data['categories']:
            if category_info['name'] == class_name:
                category_ids.append(category_info['id'])
    return category_ids


def get_preprocessed_sample(config, image_id, data, image):
    """Get image and class masks for one sample.
    """
    img_h, img_w = image.shape[:2]
    new_img_h, new_img_w = \
        config.get_image('height'), config.get_image('width')
    image = cv2.resize(image, (new_img_w, new_img_h), cv2.INTER_AREA)

    # get class masks for sample
    class_masks = []
    for class_name, params in config.get_classes().items():
        categories_ids = class_names2id(params['annotation_classes'], data)
        polygons = get_class_polygons(image_id, data, image, categories_ids)
        polygons = polygon_resize(polygons, img_h, img_w, new_img_w, new_img_h)
        # convert polygon to mask
        mask = polygons
        for process_name, process_args in params['polygon2mask'].items():
            mask = PREPROCESS_FUNC[process_name](
                mask, new_img_h, new_img_w, **process_args)
        class_masks.append(mask)

    # stack class masks to target
    target = np.stack(class_masks, -1)
    return image, target


def preprocess_data(config, json_path, image_root, save_data_path):
    """Create and save targets for Unet training."""
    image_folder = Path('images')
    target_folder = Path('targets')
    # create folders
    save_root = Path(save_data_path).parent
    image_dir = save_root / image_folder
    os.makedirs(str(image_dir), exist_ok=True)
    target_dir = save_root / target_folder
    os.makedirs(str(target_dir), exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_paths = []
    target_paths = []
    for data_img in tqdm(data['images']):
        img_name = data_img['file_name']
        image_id = data_img['id']
        image = cv2.imread(os.path.join(image_root, img_name))
        image, target = get_preprocessed_sample(config, image_id, data, image)
        # save image and target
        cv2.imwrite(str(image_dir / img_name), image)
        image_paths.append(image_folder / img_name)
        np.save(target_dir / Path(img_name).with_suffix('.npy'), target)
        target_paths.append(target_folder / Path(img_name).with_suffix('.npy'))

    pd_data = pd.DataFrame(
        list(zip(image_paths, target_paths)),
        columns=['image', 'target']
    )
    pd_data.to_csv(save_data_path, index=False)


def main(args):
    config = Config(args.config_path)
    for dataset in config.get_train('datasets'):
        preprocess_data(
            config=config,
            json_path=dataset['json_path'],
            image_root=dataset['image_root'],
            save_data_path=dataset['processed_data_path']
        )
    for dataset in config.get_val('datasets'):
        preprocess_data(
            config=config,
            json_path=dataset['json_path'],
            image_root=dataset['image_root'],
            save_data_path=dataset['processed_data_path']
        )
    for dataset in config.get_test('datasets'):
        preprocess_data(
            config=config,
            json_path=dataset['json_path'],
            image_root=dataset['image_root'],
            save_data_path=dataset['processed_data_path']
        )


PREPROCESS_FUNC = {
    "ShrinkMaskMaker": get_shrink_mask,
    "BorderMaskMaker": get_border_mask
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/scripts/segm_config.json',
                        help='Path to config.json.')
    args = parser.parse_args()
    main(args)
