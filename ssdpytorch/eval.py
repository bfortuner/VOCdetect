"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = VOCroot + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

"""
    Returns array of GT BBs for a given image
        {
            'name': label_name,
            'bbox': [
                [xmin, ymin, xmax, ymax],
                [xmin, ymin, xmax, ymax],
            ],
            ...
        }
"""

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):

    # For each class, write a separate file
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:

            # For all images in dataset
            for im_ind, index in enumerate(dataset.ids):

                # Get N class detections for img_id
                # Since there can be multiple boxes of same class predicted
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                
                # For each detection, formate and write line to file
                # img_id, conf, xmin, ymin, xmax, ymax
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

"""
        class1_GT = {
        img_1: {
            'bbox': [
                [xmin, ymin, xmax, ymax],
                [xmin, ymin, xmax, ymax],
            ],
            'det': [
                False, # used to indicate we matched a detection to this box already so don't do it again
                False
            ]
        }
    }
"""

def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # For each class
    for i, cls in enumerate(labelmap):

        # Load the class detections file (model predictions)
        filename = get_voc_results_file_template(set_type, cls)

        # Evaluate recall, precision, AP for this class/label
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')


    # read list of images - separate img_id file since we don't have access to dataset.ids anymore
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()

    # Getting the img_ids
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):

        # load annots
        recs = {}

        #for each img_id get the cooresponding GT boxes (all classes)
        # build a helper object `rects`
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # get the GT boxes specific to this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:

        # Store list of GT BBs objects for this classname in `R`
        R = [obj for obj in recs[imagename] if obj['name'] == classname]

        # Extract the BBoxes List from each BoxObj
        bbox = np.array([x['bbox'] for x in R])

        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        
        # Store boolean for each Box stating whether we've matched it yet
        # This is used to avoid double-predicting on the same GT, instead we take the most confident
        det = [False] * len(R)

        # Remove difficult boxes from the total detection count?
        npos = npos + sum(~difficult)

        # Store new class specific GT dictionary for every image with
        # only the GT BBs matching this class
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    """
    class1_GT = {
        img_1: {
            'bbox': [
                [xmin, ymin, xmax, ymax],
                [xmin, ymin, xmax, ymax],
            ],
            'det': [
                False, # used to indicate we matched a detection to this box already so don't do it again
                False
            ]
        }
    }
    """

    # read dets
    # img_ids may show up multiple times in this file (once for each box detected)
    detfile = detpath.format(classname)

    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        # detection (many to one)
        # img_id, conf, xmin, ymin, xmax, ymax
        # img_id, conf, xmin, ymin, xmax, ymax

        # Get image_id array
        image_ids = [x[0] for x in splitlines]
        
        # Get confidence array
        confidence = np.array([float(x[1]) for x in splitlines])

        # Get BB array (importantly, a numpy matrix)
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence (default is ASC, so we negate)
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)

        # Sort the BBs and Ids by confidence
        # We are going to use this for "duplicate" detection
        # If our model predictions similar boxes for the same GT
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids) # These are actually detections, since image ids can show up multiple times
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        # Loop through the image ids indexs (ranked by confidence)
        for d in range(nd):

            # Get the GT box annotation object
            R = class_recs[image_ids[d]]

            # Get the Model Detected BBs (index, then N rows of different boxes)
            bb = BB[d, :].astype(float)
            """
            bb = np.array([x1,y1,x2,y2])
            """

            ovmax = -np.inf

            # Get the GT BBs for this image (and class)
            BBGT = R['bbox'].astype(float)
            """
            BBGT = np.array([
                [x1,y1,x2,y2],
                [x1,y1,x2,y2],
                [x1,y1,x2,y2]
            ])

            (# GTs x 4)
            """


            if BBGT.size > 0:
                # compute overlaps
                # intersection
                # This is a normal IoU calculation, except
                # We calculate the overlap between a single model
                # predicted detection BB and ALL the GT BBs present in
                # the image
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])

                # What comes out of this is is an array
                # of min/max x/y values for every GT box 
                # compared to the single detected box
                """
                ixmin = np.array([
                    [b1_min]
                    [b2_min]
                    [b3_min]
                ])
                Is it flat??
                """

                # Get the width and height
                # of the intersection with every box
                # with 0 as the min
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)

                # Calculate the Area of the intersection
                inters = iw * ih
                """
                inter = np.array([
                    [I1]
                    [I2]
                    [I3]
                ])
                """

                # Union
                # Calculate the area of the two boxes
                 #and subtract the intersection
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)

                # Calculate the overlap/ IoU for each
                # GT + bb pair
                overlaps = inters / uni

                """
                overlaps = np.array([
                    [O1]
                    [O2]
                    [O3]
                ])
                """
                
                # Get the GT box with the MAX overlap
                # This is how we filter out duplicates
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps) # max idx

            # If our best match is > required threshold, 
            if ovmax > ovthresh:

                # And it's not difficult
                if not R['difficult'][jmax]:

                    # If we haven't already matched it
                    # It's a true positive
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    # Otherwise, it's a FP since
                    # we've already matched it with a higher-scoring box
                    # This is because there are multiple detections for the same image
                    # and same class. We've ranked them by confidence and mapped them to their GT 
                    # partner, if we get to a box that's already been matched (and yet we are 
                    # trying to match again), then this gets penalized to avoid duplicates
                    else:
                        fp[d] = 1.

            else:
                # If the overlap to the closest matching GT box 
                # is less then our minimum required threshold, this
                # is a FP and we shouldn't have predicted this box
                # in this location? or at all? both?
                fp[d] = 1.

        # compute precision recall
        # This cumulative sum function adds all previous values
        # to the current value for the length of the array
        # It's used for the AP calculation, and has something to do with
        # prioritization/ranking, we weight the earlier samples more importantly?
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        # I don't understand this metric exactly....
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection(args.voc_root, [('2007', set_type)], 
    BaseTransform(300, dataset_mean), AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
