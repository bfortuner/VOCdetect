import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import config as cfg
import constants as c



def predict(model, img, orig_dims, detect_fn, thresh):
    model.eval()
    img = Variable(img.cuda(async=True))
    out = model(img)
    detections = detect_fn(
        out[0],
        nn.Softmax()(out[1].view(-1, 21)),
        out[2].type(type(out[2].data))
    ).data
    w = orig_dims[0]['w']
    h = orig_dims[0]['h']
    scale = torch.Tensor([w, h, w, h])
    bboxes = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= thresh:
            score = detections[0,i,j,0]
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            label = cfg.IDX_TO_LABEL[i-1]
            bboxes.append({
                'label': label,
                'score': float(score),
                'xmin': float(pt[0]),
                'ymin': float(pt[1]),
                'xmax': float(pt[2]+1),
                'ymax': float(pt[3]+1)
            })
            j += 1

    return bboxes


def get_predictions(model, loader, detect_fn, thresh):
    preds = []
    model.eval()
    for img, targs, dims, idx in loader:
        bboxes = predict(model, img, dims, detect_fn, thresh)
        preds.append({
            'img_id': loader.dataset.img_ids[idx[0]],
            'bboxes': bboxes
        })

    return preds