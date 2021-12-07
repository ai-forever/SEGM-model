import numpy as np
import json
import cv2
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import argparse

from segm.src.config import Config
from segm.src.dataset import MakeShrinkMask, MakeBorderMask


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [[list_of_numbers[i], list_of_numbers[i+1]]
            for i in range(0, len(list_of_numbers), 2)]
    return np.array(bbox)


def get_shrink_mask(polygons, image_h, image_w, shrink_ratio):
    shrink_mask_maker = MakeShrinkMask(image_h, image_w, shrink_ratio)
    for polygon in polygons:
        shrink_mask_maker.add_polygon_to_mask(polygon)
    return shrink_mask_maker.get_shrink_mask()


def get_border_mask(polygons, image_h, image_w, shrink_ratio):
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


def get_preprocess_sample(config, image_id, data, image, category_ids):
    polygons = []
    for data_ann in data['annotations']:
        if (
            data_ann['image_id'] == image_id
            and data_ann['category_id'] in category_ids
            and data_ann['segmentation']
        ):
            polygon = numbers2coords(data_ann['segmentation'][0])
            polygons.append(polygon)

    img_h, img_w = image.shape[:2]
    new_img_w, new_img_h = config.get_image('height'), config.get_image('width')
    image = cv2.resize(image, (new_img_w, new_img_h), cv2.INTER_AREA)
    polygons = polygon_resize(polygons, img_h, img_w, new_img_w, new_img_h)

    img_h, img_w = image.shape[:2]
    shrink_mask = get_shrink_mask(polygons, img_h, img_w,
                                  config.get('mask_shrink_ratio'))
    border_mask = get_border_mask(polygons, img_h, img_w,
                                  config.get('border_shrink_ratio'))
    return image, shrink_mask, border_mask


def preprocess_data(
    config, json_path, image_root, category_ids, save_data_path
):
    """Create and save targets for DBnet training (shrink and border masks).
    """
    image_folder = Path('images')
    shrink_folder = Path('shrink_targets')
    border_folder = Path('border_targets')

    # create folders
    save_root = Path(save_data_path).parent
    image_dir = save_root / image_folder
    os.makedirs(str(image_dir), exist_ok=True)
    shrink_dir = save_root / shrink_folder
    os.makedirs(str(shrink_dir), exist_ok=True)
    border_dir = save_root / border_folder
    os.makedirs(str(border_dir), exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_paths = []
    shrink_paths = []
    border_paths = []
    for data_img in tqdm(data['images']):
        img_name = data_img['file_name']
        image_id = data_img['id']
        image = cv2.imread(os.path.join(image_root, img_name))
        image, shrink_mask, border_mask = get_preprocess_sample(
            config, image_id, data, image, category_ids)
        # save image
        cv2.imwrite(str(image_dir / img_name), image)
        image_paths.append(image_folder / img_name)
        # save shrink mask
        np.save(shrink_dir / Path(img_name).with_suffix('.npy'), shrink_mask)
        shrink_paths.append(shrink_folder / Path(img_name).with_suffix('.npy'))
        # save border mask
        np.save(border_dir / Path(img_name).with_suffix('.npy'), border_mask)
        border_paths.append(border_folder / Path(img_name).with_suffix('.npy'))

    pd_data = pd.DataFrame(
        list(zip(image_paths, shrink_paths, border_paths)),
        columns=['file_name', 'srink_mask_name', 'border_mask_name']
    )
    pd_data.to_csv(save_data_path, index=False)


def main(args):
    config = Config(args.config_path)
    preprocess_data(
        config=config,
        json_path=config.get_train('json_path'),
        image_root=config.get_train('image_root'),
        category_ids=config.get_train('category_ids'),
        save_data_path=config.get_train('processed_data_path')
    )
    preprocess_data(
        config=config,
        json_path=config.get_val('json_path'),
        image_root=config.get_val('image_root'),
        category_ids=config.get_val('category_ids'),
        save_data_path=config.get_val('processed_data_path')
    )
    preprocess_data(
        config=config,
        json_path=config.get_test('json_path'),
        image_root=config.get_test('image_root'),
        category_ids=config.get_test('category_ids'),
        save_data_path=config.get_test('processed_data_path')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='/workdir/segm/config.json',
                        help='Path to config.json.')
    args = parser.parse_args()
    main(args)
