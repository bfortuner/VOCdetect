import os
import numpy as np
import pandas as pd

import init_project
import constants as c
import datasets.metadata as meta
import utils

PROJECT_NAME = 'udacity'
PROJECT_PATH = '/bigguy/data/' + PROJECT_NAME
PROJECT_TYPE = c.OBJECT_DETECTION
IMG_INPUT_FORMATS = [c.JPG]
IMG_TARGET_FORMATS = []
IMG_DATASET_TYPES = [c.TRAIN, c.TEST]
PATHS = init_project.init_paths(PROJECT_PATH, IMG_DATASET_TYPES,
    IMG_INPUT_FORMATS, IMG_TARGET_FORMATS)
CROWD_AI_METADATA_PATH = os.path.join(PROJECT_PATH, 'labels_crowdai.csv')

AUTTI_LABELS_PATH = os.path.join(PROJECT_PATH, 'labels_autti.csv')
AUTTI_IMG_PATH = os.path.join(PROJECT_PATH, 'autti')
AUTTI_METADATA_PATH = os.path.join(PROJECT_PATH, 'metadata_autti.json')

LABEL_NAMES = (
    'truck', 'biker', 'trafficlight', 'car', 'pedestrian')
ATTRIBUTES = ()
LABEL_TO_IDX = meta.get_labels_to_idxs(LABEL_NAMES)
IDX_TO_LABEL = meta.get_idxs_to_labels(LABEL_NAMES)


def make_fold(meta, name, val_pct):
    img_ids = list(meta['imgs'].keys())
    val_size = int(len(img_ids)*val_pct)
    fold = {
        c.TRAIN: img_ids[:-val_size],
        c.VAL: img_ids[-val_size:],
        c.TEST: []
    }
    fpath = os.path.join(PATHS['folds'], name+'.json')
    utils.files.save_json(fpath, fold)
    return fold

"""
1478019952686311006.jpg 950 574 1004 620 0 "car"
1478019952686311006.jpg 1748 482 1818 744 0 "pedestrian"
"""

def make_autti_meta_file():
    meta = {
        'imgs': {},
        'img_ext': c.JPG_EXT,
        'img_dir': AUTTI_IMG_PATH
    }
    ids_ = set()
    with open(AUTTI_LABELS_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip('\n').split(' ')
            img_id = data[0].strip(c.JPG_EXT)
            ids_.add(img_id)
            bbox = {
                'label': data[6].strip('"').lower(),
                'xmin': int(data[1]),
                'ymin': int(data[2]),
                'xmax': int(data[3]),
                'ymax': int(data[4]),
                'occluded': int(data[5]),
                'attributes': None if len(data) < 8 else data[7].strip('"').lower(),
                'difficult': False
            }
            if img_id in meta['imgs']:
                meta['imgs'][img_id]['bboxes'].append(bbox)                
            else: 
                meta['imgs'][img_id] = {
                    'img_id': img_id,
                    'bboxes': [bbox]
                }
    utils.files.save_json(AUTTI_METADATA_PATH, meta)
    return meta, ids_


"""
CrowdAI
xmin,ymin,xmax,ymax,Frame,Label,Preview URL
785,533,905,644,1479498371963069978.jpg,Car,http://crowdai.com/images/Wwj-gorOCisE7uxA/visualize
89,551,291,680,1479498371963069978.jpg,Car,http://crowdai.com/images/Wwj-gorOCisE7uxA/visualize
"""

def make_crowd_ai_file(inpath, dset_name, img_dir):
    meta = {
        'imgs': {},
        'img_ext': c.JPG_EXT,
        'img_dir': img_dir
    }
    with open(inpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            img_id = data[4].strip(c.JPG_EXT)
            bbox = {
                'label': data[5].lower(),
                'xmin': int(data[0]),
                'ymin': int(data[1]),
                'xmax': int(data[2]),
                'ymax': int(data[3]),
            }
            if img_id in meta['imgs']:
                meta['imgs'][img_id]['bboxes'].append(bbox)                
            else: 
                meta['imgs'][img_id] = {
                    'img_id': img_id,
                    'bboxes': [bbox]
                }
    meta_fpath = os.path.join(PATHS['project'],
                              'metadata_{:s}.json'.format(dset_name))
    utils.files.save_json(meta_fpath, meta)
    return meta


def load_metadata(dset_name):
    meta_fpath = os.path.join(PATHS['project'], 
                              'metadata_{:s}.json'.format(dset_name))
    return utils.files.load_json(meta_fpath)
