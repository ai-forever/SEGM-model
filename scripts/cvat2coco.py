import argparse
import json
import xml.etree.ElementTree as ET


def points_str2list(raw_points):
    """Convert points in string format to list of floats."""
    points_float = []
    for xy_points in raw_points.split(';'):
        x, y = xy_points.split(',')
        points_float.append(float(x))
        points_float.append(float(y))
    return [points_float]


def class_names2id(class_name, categories_data):
    """Match class names to categoty ids using annotation in COCO format."""
    for category_info in categories_data:
        if category_info['name'] == class_name:
            return category_info['id']
    return None


def cvat_xml2coco_json(cvat_xml_path):
    root = ET.parse(cvat_xml_path).getroot()
    categories = []
    for idx, data in enumerate(root.findall('meta/task/labels/label/name')):
        categories.append(
            {
                'id': idx,
                'name': data.text
            }

        )

    images = []
    annotations = []
    for image_data in root.findall('image'):
        image = {}
        image['height'] = int(image_data.attrib['height'])
        image['width'] = int(image_data.attrib['width'])
        image['id'] = int(image_data.attrib['id'])
        image['file_name'] = image_data.attrib['name']
        images.append(image)

        for polygon_data in image_data.findall('polygon'):
            annotation = {}
            annotation['category_id'] = class_names2id(
                polygon_data.attrib['label'],
                categories
            )
            annotation['image_id'] = int(image_data.attrib['id'])
            if polygon_data.get('group_id') is not None:
                annotation['group_id'] = int(polygon_data.attrib['group_id'])
            annotation['segmentation'] = points_str2list(
                polygon_data.attrib['points'])
            for attribute_data in polygon_data.findall('attribute'):
                annotation['attributes'] = {
                    'occluded': False if polygon_data.attrib['occluded'] == '0' else True,
                    'translation': attribute_data.text
                }
            annotations.append(annotation)

        for polyline_data in image_data.findall('polyline'):
            annotation = {}
            annotation['category_id'] = class_names2id(
                polyline_data.attrib['label'],
                categories
            )
            annotation['image_id'] = int(image_data.attrib['id'])
            annotation['segmentation'] = points_str2list(
                polyline_data.attrib['points'])
            annotations.append(annotation)

    return {
        'categories': categories,
        'images': images,
        'annotations': annotations
    }


def main(args):
    data = cvat_xml2coco_json(args.cvat_xml_path)
    with open(args.coco_json_save_path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvat_xml_path', type=str, required=True,
                        help='Path to cvat annotations in CVAT xml format.')
    parser.add_argument('--coco_json_save_path', type=str, required=True,
                        help='Path to save the output COCO json.')
    args = parser.parse_args()

    main(args)
