import os

from tensorpack.utils import logger

from dataset import DatasetRegistry
from dataset.coco import COCODetection


__all__ = ['register_open_images']


class OpenImagesDetection(COCODetection):
    def __init__(self, basedir, split):
        self._basedir = basedir
        super().__init__(basedir=basedir, split=split, test_seg=False)

    def get_images_dir(self, split):
        return os.path.realpath(os.path.join(self._basedir, 'images', split, 'original'))

    def get_annotations_file(self, split):
        return os.path.join(self._basedir, 'coco_annotations/instances_{}_2019.json'.format(split))


def register_open_images(basedir):

    class_desc_file = os.path.join(basedir, 'docs/challenge-2019-classes-description-500.csv')

    if os.path.exists(class_desc_file):
        logger.info('Registering Open Images dataset')

        class_desc = [l.strip().split(',') for l in open(class_desc_file)]
        class_names = [l[1] for l in class_desc]
        class_names = ["BG"] + class_names

        for split in ["train", "validation"]:
            name = "open_images_" + split
            DatasetRegistry.register(name, lambda x=split: OpenImagesDetection(basedir, x))
            DatasetRegistry.register_metadata(name, 'class_names', class_names)


if __name__ == '__main__':
    basedir = r'X:\open_images_v5'
    c = OpenImagesDetection(basedir, 'validation')
    roidb = c.load(add_gt=True, add_mask=False)
    print("#Images:", len(roidb))
