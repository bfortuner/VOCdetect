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
PATHS = init_project.init_paths(PROJECT_PATH, IMG_DATASET_TYPES,
    IMG_INPUT_FORMATS, IMG_TARGET_FORMATS)
VOC_DEV_KIT_PATH = os.path.join(PROJECT_PATH, 'VOCdevkit')

LABEL_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

LABEL_TO_IDX = meta.get_labels_to_idxs(LABEL_NAMES)
IDX_TO_LABEL = meta.get_idxs_to_labels(LABEL_NAMES)


def get_voc_root(year):
    return os.path.join(VOC_DEV_KIT_PATH, 'VOC'+year)


def make_2007_fold():
    dir_path = os.path.join(get_voc_root('2007'), 'ImageSets', 'Main')
    fold = {
        c.TRAIN: get_img_ids_from_file(os.path.join(dir_path, 'train.txt')),
        c.VAL: get_img_ids_from_file(os.path.join(dir_path, 'val.txt')),
        c.TEST: get_img_ids_from_file(os.path.join(dir_path, 'test.txt'))
    }     
    fold_fpath = os.path.join(PATHS['folds'], '2007.json')
    utils.files.save_json(fold_fpath, fold)
    return fold


def make_2012_fold():
    dir_path = os.path.join(get_voc_root('2012'), 'ImageSets', 'Main')
    fold = {
        c.TRAIN: get_img_ids_from_file(os.path.join(dir_path, 'train.txt')),
        c.VAL: get_img_ids_from_file(os.path.join(dir_path, 'val.txt')),
        c.TEST: []
    }     
    fold_fpath = os.path.join(PATHS['folds'], '2012.json')
    utils.files.save_json(fold_fpath, fold)
    return fold


def get_img_ids_from_file(fpath):
    img_ids = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_ids.append(line.split()[0])
    return img_ids


def make_metadata_file(year):
    annot_path = os.path.join(get_voc_root(year), 'Annotations')
    img_path = os.path.join(get_voc_root(year), 'JPEGImages')
    fpaths, fnames = utils.files.get_paths_to_files(
        annot_path, file_ext=c.XML_EXT, strip_ext=True)
    meta = {
        'imgs': {},
        'img_ext': c.JPG_EXT,
        'img_dir': img_path
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
                    'ymax': int(float(bbox.find('ymax').text)),
                    'difficult': int(child.find('difficult').text)
                })
        meta['imgs'][fnames[i]] = {
            'img_id': fnames[i],
            'bboxes': bboxes
        }
    meta_fpath = os.path.join(PATHS['project'], 
                              'metadata_{:s}.json'.format(year))
    utils.files.save_json(meta_fpath, meta)
    return meta

def load_metadata(year):
    meta_fpath = os.path.join(PATHS['project'], 
                              'metadata_{:s}.json'.format(year))
    return utils.files.load_json(meta_fpath)
