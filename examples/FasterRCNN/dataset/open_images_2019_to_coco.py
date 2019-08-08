import os
import json
import argparse

import cv2
import pandas as pd
from tqdm import tqdm


def read_class_descriptions(path):
    print('Loading class descriptions...')
    content = [l.strip().split(',') for l in open(path)]
    class_id_to_index = {a[0]: i + 1 for i, a in enumerate(content)}
    class_id_to_name = {a[0]: a[1] for a in content}
    return content, class_id_to_index, class_id_to_name


def add_class_descriptions_to_json(class_descriptions, train_annotations, val_annotations):
    datas = [train_annotations, val_annotations]
    for d in datas:
        d['categories'] = []
    for i, a in enumerate(class_descriptions):
        for d in datas:
            d['categories'] += [{'supercategory': 'none', 'id': i + 1, 'name': a[1]}]


def convert_annotations(data, path, class_id_to_index, images_root_dir, output_dir):
    df = pd.read_csv(path)

    data['images'] = []
    data['annotations'] = []

    image_ids = set()
    failed_image_ids = set()

    image_id_to_shape = {}

    _set = data['info']['set']

    for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing {} set'.format(_set)):

        image_id = row['ImageID']

        if image_id in failed_image_ids:
            continue

        if image_id not in image_ids:
            image_path = os.path.join(images_root_dir, _set, 'original', image_id + '.jpg')
            image = cv2.imread(image_path)
            if image is None:
                failed_image_ids.add(image_id)
                continue
            h, w = image.shape[:2]
            image_id_to_shape[image_id] = (h, w)
            data['images'] += [{'id': image_id, 'file_name': image_id + '.jpg', 'height': h, 'width': w}]

        h, w = image_id_to_shape[image_id]
        x_min, y_min, x_max, y_max = row['XMin'] * w, row['YMin'] * h, row['XMax'] * w, row['YMax'] * h
        area = (x_max - x_min) * (y_max - y_min)

        data['annotations'] += [{
            'id': index,
            'area': area,
            'iscrowd': row['IsGroupOf'],
            'image_id': image_id,
            'bbox': [x_min, y_min, (x_max - x_min), (y_max - y_min)],
            'category_id': class_id_to_index[row['LabelName']]
        }]

    output_path = os.path.join(output_dir, 'instances_{}_2019.json'.format(_set))
    with open(output_path, 'w') as f:
        json.dump(data, f)


def main(args):

    class_descriptions, class_id_to_index, class_id_to_name = read_class_descriptions(args.class_descriptions)

    train_annotations = {
        'info': {'description': 'Open Images Object Detection 2019 - Train Set', 'year': 2019, 'set': 'train'}
    }
    val_annotations = {
        'info': {'description': 'Open Images Object Detection 2019 - Validation Set', 'year': 2019, 'set': 'validation'}
    }

    add_class_descriptions_to_json(class_descriptions, train_annotations, val_annotations)

    convert_annotations(data=val_annotations,
                        path=args.open_images_val_annotations,
                        class_id_to_index=class_id_to_index,
                        images_root_dir=args.images_root_dir,
                        output_dir=args.output_dir)

    convert_annotations(data=train_annotations,
                        path=args.open_images_train_annotations,
                        class_id_to_index=class_id_to_index,
                        images_root_dir=args.images_root_dir,
                        output_dir=args.output_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--class_descriptions',
                   default=r'X:\open_images_v5\docs\challenge-2019-classes-description-500.csv')
    p.add_argument('--open_images_train_annotations',
                   default=r'X:\open_images_v5\docs\challenge-2019-train-detection-bbox.csv')
    p.add_argument('--open_images_val_annotations',
                   default=r'X:\open_images_v5\docs\challenge-2019-validation-detection-bbox.csv')
    p.add_argument('--images_root_dir',
                   default=r'X:\open_images_v5\images')
    p.add_argument('--output_dir',
                   default=r'X:\open_images_v5\coco_annotations')
    main(p.parse_args())
