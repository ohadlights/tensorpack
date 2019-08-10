import os

from tensorpack.utils import logger

from config import config as cfg
from dataset import DatasetRegistry, DatasetSplit
from dataset.coco import COCODetection


__all__ = ['register_open_images']


# class OpenImagesDetection(DatasetSplit):
#
#     _IMAGES_FOLDER = {'train2019': 'train', 'val2019': 'validation'}
#     _ANNOTATIONS_FILE = {'train2019': 'challenge-2019-train-detection-bbox.csv',
#                          'val2019': 'challenge-2019-validation-detection-bbox.csv'}
#
#     def __init__(self, basedir, split):
#         basedir = os.path.expanduser(basedir)
#
#         self._imgdir = os.path.realpath(os.path.join(basedir, 'images', self._IMAGES_FOLDER[split], 'original'))
#         assert os.path.isdir(self._imgdir), "{} is not a directory!".format(self._imgdir)
#
#         class_desc_file = os.path.join(basedir, 'docs/challenge-2019-classes-description-500.csv')
#         assert os.path.isfile(class_desc_file), class_desc_file
#
#         annotation_file = os.path.join(basedir, 'docs', self._ANNOTATIONS_FILE[split])
#         assert os.path.isfile(annotation_file), annotation_file
#
#         class_desc = [l.strip().split(',') for l in open(class_desc_file)]
#         self._class_names = [l[1] for l in class_desc]
#         self._class_id_to_name = {l[0]: l[1] for l in class_desc}
#         cfg.DATA.CLASS_NAMES = ["BG"] + self._class_names
#
#         # TODO: parse annotations file
#         content = [l.strip().split(',') for l in open(annotation_file).readlines()[1:]]
#         logger.info("Instances loaded from {}.".format(annotation_file))


class OpenImagesDetection(COCODetection):
    def __init__(self, basedir, split):
        self._basedir = basedir

        super().__init__(basedir=basedir, split=split)

        class_desc_file = os.path.join(basedir, 'docs/challenge-2019-classes-description-500.csv')
        assert os.path.isfile(class_desc_file), class_desc_file

        class_desc = [l.strip().split(',') for l in open(class_desc_file)]
        self.class_names = [l[1] for l in class_desc]
        cfg.DATA.CLASS_NAMES = ["BG"] + self.class_names

    def get_images_dir(self, split):
        return os.path.realpath(os.path.join(self._basedir, 'images', split, 'original'))

    def get_annotations_file(self, split):
        return os.path.join(self._basedir, 'coco_annotations/instances_{}_2019.json'.format(split))


def register_open_images(basedir):
    for split in ["train", "validation"]:
        DatasetRegistry.register("open_images_" + split, lambda x=split: OpenImagesDetection(basedir, x))


if __name__ == '__main__':
    basedir = r'X:\open_images_v5'
    c = OpenImagesDetection(basedir, 'validation')
    roidb = c.load(add_gt=True, add_mask=False)
    print("#Images:", len(roidb))
