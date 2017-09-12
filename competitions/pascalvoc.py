import os
import numpy as np
import pandas as pd
import init_project
import constants as c
from xml.etree import ElementTree

import datasets.metadata as meta
import utils

PROJECT_NAME = 'pascalvoc'
PROJECT_PATH = '/bigguy/data/' + PROJECT_NAME
PROJECT_TYPE = c.OBJECT_DETECTION
IMG_INPUT_FORMATS = [c.JPG]
IMG_TARGET_FORMATS = [c.XML, c.PNG]
IMG_DATASET_TYPES = [c.TRAIN, c.TEST]
METADATA_PATH = os.path.join(PROJECT_PATH, 'metadata.json')
PATHS = init_project.init_paths(PROJECT_PATH, IMG_DATASET_TYPES,
    IMG_INPUT_FORMATS, IMG_TARGET_FORMATS)
XML_PATH = PATHS['datasets']['targets']['trn_xml']

LABEL_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

LABEL_TO_IDX = meta.get_labels_to_idxs(LABEL_NAMES)
IDX_TO_LABEL = meta.get_idxs_to_labels(LABEL_NAMES)


def make_metadata_file():
    fpaths, fnames = utils.files.get_paths_to_files(
        XML_PATH, file_ext=c.XML_EXT, strip_ext=True)
    meta = {
        'imgs': {},
        'img_ext': c.JPG_EXT
    }
    for i in range(len(fpaths)):
        tree = ElementTree.parse(fpaths[i])
        bboxes = []
        for child in tree.getroot():
            if child.tag == 'object':
                bbox = child.find('bndbox')
                bboxes.append({
                    'label': child.find('name').text,
                    'xmin': int(float(bbox.find('xmin').text)),
                    'ymin': int(float(bbox.find('ymin').text)),
                    'xmax': int(float(bbox.find('xmax').text)),
                    'ymax': int(float(bbox.find('ymax').text))
                })
        meta['imgs'][fnames[i]] = {
            'img_id': fnames[i],
            'bboxes': bboxes
        }
    utils.files.save_json(METADATA_PATH, meta)

