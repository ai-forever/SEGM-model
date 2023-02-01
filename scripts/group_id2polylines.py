import json
import cv2
from tqdm import tqdm
import numpy as np
import argparse


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [[int(list_of_numbers[i]), int(list_of_numbers[i+1])]
            for i in range(0, len(list_of_numbers), 2)]
    return np.array(bbox)


def coord2numbers(polygon):
    numbers = [coord for coords in polygon for coord in coords]
    return [numbers]


def get_polygons_by_group_id(data, image_id, group_id):
    """Get polygons from annotation for a specific image and group_id.
    """
    polygons = []
    for data_ann in data['annotations']:
        if (
            data_ann['image_id'] == image_id
            and data_ann.get('group_id') == group_id
            and data_ann['segmentation']
        ):
            polygon = numbers2coords(data_ann['segmentation'][0])
            polygons.append(polygon)
    return polygons


def get_group_ids_for_image(image_id, data):
    """Get group indexesfor a specific image."""
    group_ids = set()
    for data_ann in data['annotations']:
        if (
            data_ann['image_id'] == image_id
            and 'group_id' in data_ann
        ):
            group_ids.add(data_ann['group_id'])
    return group_ids


def merge_polygins_to_line(polygons):
    line = []
    for polygon in polygons:
        M = cv2.moments(np.array([polygon]))
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            line.append([cX, cY])
        else:
            print(polygon)
    if len(line) == 1:
        x_min = polygons[0].min(0)[0]
        x_max = polygons[0].max(0)[0]
        w = x_max - x_min
        left_dot = [line[0][0] - w/4, line[0][1]]
        right_dot = [line[0][0] + w/4, line[0][1]]
        line.append(left_dot)
        line.append(right_dot)
    line = sorted(line, key=lambda tup: tup[0])
    return line


def add_new_category_id(data, category_name):
    max_category_index = 0
    for data_category in data['categories']:
        if data_category['id'] > max_category_index:
            max_category_index = data_category['id']

    new_category_index = max_category_index + 1
    data['categories'].append(
        {
            'id': new_category_index,
            'name': category_name
        }
    )
    return new_category_index


def main(args, line_name='text_line'):
    with open(args.annotation_json_path, 'r') as f:
        data = json.load(f)

    text_line_category_id = add_new_category_id(data, line_name)

    for data_img in tqdm(data['images']):
        image_id = data_img['id']
        group_ids = get_group_ids_for_image(image_id, data)
        for group_id in group_ids:
            polygons = get_polygons_by_group_id(data, image_id, group_id)
            polyline = merge_polygins_to_line(polygons)
            polyline = coord2numbers(polyline)
            data['annotations'].append(
                {
                    'category_id': text_line_category_id,
                    'image_id': image_id,
                    'segmentation': polyline
                }
            )

    with open(args.annotation_save_path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_json_path', type=str,
                        default='./annotation.json',
                        help='Path to annotation.json')
    parser.add_argument('--annotation_save_path', type=str,
                        default='./annotation-polylines.json',
                        help='Path to save json with new polylines')
    args = parser.parse_args()
    main(args)
