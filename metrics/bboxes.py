import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import metric_utils


def get_preds_by_class(preds, label_names):
    '''
    Input = [
        'img_id': 9879827,
        'bboxes': [{
            'label': label,
            'score': float(score),
            'xmin': float(pt[0]),
            'ymin': float(pt[1]),
            'xmax': float(pt[2]+1),
            'ymax': float(pt[3]+1)
        }]
    ]
    #image ids can show up multiple times in same label list
    Returns {
        label : [
            [img_id, conf, xmin, ymin, xmax, ymax],
            [img_id, conf, xmin, ymin, xmax, ymax]
        ],
        label2 : [
            [img_id, conf, xmin, ymin, xmax, ymax],
            [img_id, conf, xmin, ymin, xmax, ymax]        
        ]
    }
    '''
    class_preds = {
        lb: [] for lb in label_names
    }
    for pred in preds:
        for bb in pred['bboxes']:
            label = bb['label']
            img_id = pred['img_id']
            class_preds[label].append(
                [img_id, bb['score'], bb['xmin'], \
                    bb['ymin'], bb['xmax'], bb['ymax']]
            )
    return class_preds


def bb_to_list(bb):
    return [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]


def get_targs_by_class(targs, label):
    # get the GT boxes specific to this class
    label_bbs = {}
    n_bbs = 0
    for img_id in targs.keys():
        # Store list of GT BBs objects for this label
        img_bbs = [bb for bb in targs[img_id]['bboxes'] \
                    if bb['label'] == label]
        
        # Extract the BBoxes List from each BoxObj
        bboxes = np.array(
            [bb_to_list(bb) for bb in img_bbs]
        )
        # Store boolean for each Box stating whether we've matched it yet
        # This is used to avoid double-predicting on the same GT, instead we take the most confident
        matched = [False] * len(bboxes)
        label_bbs[img_id] = {'bboxes': bboxes,
                             'matched': matched}
        n_bbs += len(bboxes)
    
    return label_bbs, n_bbs


def get_label_level_aps(all_preds, targs, labels, thresh=0.5):
    label_preds = get_preds_by_class(
        all_preds, labels)
    label_aps = {lb: {} for lb in labels}
    for label, preds in label_preds.items():
        gt_bbs, total_gt_bbs = get_targs_by_class(
            targs, label)
        if len(preds) > 0:
            img_ids = [x[0] for x in preds]
            confidence = np.array([float(x[1]) for x in preds])
            bbs = np.array([
                [float(coord) for coord in x[2:]] for x in preds
            ])
            sorted_idx = np.argsort(-confidence)
            sorted_conf = np.sort(-confidence)

            # Sort the BBs and Ids by confidence
            # We are going to use this for "duplicate" detection
            # If our model predictions similar boxes for the same GT
            bbs = bbs[sorted_idx, :]
            img_ids = [img_ids[x] for x in sorted_idx]

            # go down dets and mark TPs and FPs
            # These are actually detections --
            # since image ids can show up multiple times
            n_bbs = len(img_ids) 
            tp = np.zeros(n_bbs)
            fp = np.zeros(n_bbs)

            for i in range(n_bbs):
                img_gt_bbs = gt_bbs[img_ids[i]]
                bb = bbs[i, :].astype(float)
                max_iou = -np.inf

                BBGT = img_gt_bbs['bboxes'].astype(float)
                """
                bbgt = np.array([
                    [x1,y1,x2,y2],
                    [x1,y1,x2,y2],
                    [x1,y1,x2,y2]
                ])
                (# GT bbs x 4)
                """
                if len(BBGT) > 0:
                    # compute overlaps (intersection)
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
                    intersection = iw * ih
                    """
                    inter = np.array([
                        [I1]
                        [I2]
                        [I3]
                    ])
                    """

                    # Union
                    # Calculate the area of the two boxes
                    # and subtract the intersection
                    bb_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    gt_area = ((BBGT[:, 2] - BBGT[:, 0]) * 
                              (BBGT[:, 3] - BBGT[:, 1]))
                    union = (bb_area + gt_area) - intersection

                    # Calculate the overlap/ IoU for each
                    # GT + bb pair
                    IoUs = intersection / union
                    """
                    IoUs = np.array([
                        [O1]
                        [O2]
                        [O3]
                    ])
                    """

                    # Get the GT box with the MAX overlap
                    # This is how we filter out duplicates
                    max_iou = np.max(IoUs)
                    max_idx = np.argmax(IoUs) # max idx

                # If our best match is > required threshold, 
                if max_iou > thresh:
                    # If we haven't already matched it
                    # It's a true positive
                    if not img_gt_bbs['matched'][max_idx]:
                        tp[i] = 1.0
                        img_gt_bbs['matched'][max_idx] = True
                    # Otherwise, it's a FP since
                    # we've already matched it with a higher-scoring box
                    # This is because there are multiple detections for the same image
                    # and same class. We've ranked them by confidence and mapped them to their GT 
                    # partner, if we get to a box that's already been matched (and yet we are 
                    # trying to match again), then this gets penalized to avoid duplicates
                    else:
                        fp[i] = 1.0
                else:
                    # If the overlap to the closest matching GT box 
                    # is less then our minimum required threshold, this
                    # is a FP and we shouldn't have predicted this box
                    # in this location? or at all? both?
                    fp[i] = 1.0

            # compute precision recall
            # This cumulative sum function adds all previous values
            # to the current value for the length of the array
            # It's used for the AP calculation, and has something to do with
            # prioritization/ranking, we weight the earlier samples more importantly?
            cum_fp = np.cumsum(fp)
            cum_tp = np.cumsum(tp)
            recall = cum_tp / float(total_gt_bbs)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)

            # I don't understand this metric exactly....
            ap = metric_utils.get_ap(recall, precision)
        else:
            ap, tp, fp = -1, -1, -1

        label_aps[label]['ap'] = ap
        label_aps[label]['tp'] = np.sum(tp)
        label_aps[label]['fp'] = np.sum(fp)
        label_aps[label]['total_gt_bbs'] = total_gt_bbs
    
    return label_aps


def plot_label_level_aps(label_aps):
    labels = list(label_aps.keys())
    aps = [label_aps[lb]['ap'] for lb in labels]
    idxs = np.arange(len(aps))
    fig, ax = plt.subplots()
    width=.5
    bars = ax.bar(idxs, aps, width=width, color='r')
    fig.set_size_inches(18.5, 10.5, forward=True)
    ax.set_ylabel('AP')
    ax.set_xlabel('Labels')
    ax.set_title('Label-level AP')
    ax.set_xticks(idxs + width / 2)
    ax.set_xticklabels(labels, rotation='vertical')
    plt.plot()


def plot_label_level_bb_counts(label_aps):
    labels = list(label_aps.keys())
    bbs = [label_aps[lb]['total_gt_bbs'] for lb in labels]
    idxs = np.arange(len(bbs))
    fig, ax = plt.subplots()
    width=.5
    _ = ax.bar(idxs, bbs, width=width, color='r')
    fig.set_size_inches(18.5, 10.5, forward=True)
    ax.set_ylabel('GTBBs')
    ax.set_xlabel('Labels')
    ax.set_title('Label-level GTBB Counts')
    ax.set_xticks(idxs + width / 2)
    ax.set_xticklabels(labels, rotation='vertical')
    plt.plot()


def get_label_level_bb_metrics(metrics_dict):
    json_ = json.dumps(metrics_dict)
    df = pd.read_json(json_)
    return df.T
