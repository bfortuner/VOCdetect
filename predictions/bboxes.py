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

    bboxes = []
    for i in range(1, detections.size(1)):
        label = cfg.IDX_TO_LABEL[i-1]
        dets = detections[0, i, :]
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
        dets = np.hstack((boxes.cpu().numpy(), 
                scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
        for j in range(dets.shape[0]):
            bboxes.append({
                'label': label,
                'score': float(dets[j][4]),
                'xmin': float(dets[j][0]),
                'ymin': float(dets[j][1]),
                'xmax': float(dets[j][2]+1),
                'ymax': float(dets[j][3]+1)
            })
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