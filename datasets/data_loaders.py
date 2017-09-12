import os
import torch.utils.data
from torch.utils.data import DataLoader
from config import *
import utils.imgs as img_utils



class MixDataLoader():
    """
    Combines batches from two data loaders.
    Useful for pseudolabeling.
    """
    def __init__(self, dl1, dl2):
        self.dl1 = dl1
        self.dl2 = dl2
        self.dl1_iter = iter(dl1)
        self.dl2_iter = iter(dl2)
        self.n = len(dl1)
        self.cur = 0

    def _reset(self):
        self.cur = 0

    def _cat_lst(self, fn1, fn2):
        return fn1 + fn2

    def _cat_tns(self, t1, t2):
        return torch.cat([t1, t2])

    def __next__(self):
        x1,y1,f1 = next(self.dl1_iter)
        x2,y2,f2 = next(self.dl2_iter)
        while self.cur < self.n:
            self.cur += 1
            return (self._cat_tns(x1,x2), self._cat_tns(y1,y2),
                    self._cat_lst(f1,f2))

    def __iter__(self):
        self.cur = 0
        self.dl1_iter = iter(self.dl1)
        self.dl2_iter = iter(self.dl2)
        return self

    def __len__(self):
        return self.n


def get_batch(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    inputs, targets, img_paths = next(iter(dataloader))
    return inputs, targets, img_paths


def get_data_loader(dset, batch_size, shuffle=False,
                    n_workers=1, pin_memory=False):
    return DataLoader(dset, batch_size, shuffle=shuffle,
                      pin_memory=pin_memory, num_workers=n_workers)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, 0)
    return imgs, targets