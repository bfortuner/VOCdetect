import torch
import os
import cv2
import numpy as np
from . import data_utils
import utils


loaders = {
    'pil': data_utils.pil_loader,
    'tns': data_utils.tensor_loader,
    'npy': data_utils.numpy_loader,
    'tif': data_utils.tif_loader,
    'io': data_utils.io_loader,
    'cv2': data_utils.cv2_loader
}


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, fpaths,
                 img_loader='pil',
                 targets=None,
                 transform=None,
                 target_transform=None):
        self.fpaths = fpaths
        self.loader = self._get_loader(img_loader)
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def _get_loader(self, loader_type):
        return loaders[loader_type]

    def _get_target(self, index):
        if self.targets is None:
            return torch.FloatTensor(1)
        target = self.targets[index]
        if self.target_transform is not None:
            return self.target_transform(target)
        return torch.FloatTensor(target)

    def _get_input(self, index):
        img_path = self.fpaths[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        img_path = self.fpaths[index]
        return input_, target, img_path

    def __len__(self):
        return len(self.fpaths)


class MultiInputDataset(FileDataset):
    def __init__(self, fpaths,
                 img_loader='pil', #'tns', 'npy'
                 targets=None,
                 other_inputs=None,
                 transform=None,
                 target_transform=None):
        super().__init__(fpaths, img_loader, targets,
                         transform, target_transform)
        self.other_inputs = other_inputs

    def _get_other_input(self, index):
        other_input = self.other_inputs[index]
        return other_input

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        other_input = self._get_other_input(index)
        img_path = self.fpaths[index]
        return input_, target, other_input, img_path


class MultiTargetDataset(FileDataset):
    def __init__(self, fpaths,
                 img_loader='pil',
                 targets=None,
                 other_targets=None,
                 transform=None,
                 target_transform=None):
        super().__init__(fpaths, img_loader, targets,
                         transform, target_transform)
        self.other_targets = other_targets

    def _get_other_target(self, index):
        if self.other_targets is None:
            return torch.FloatTensor(1)
        other_target = self.other_targets[index]
        return torch.FloatTensor(other_target)

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        other_target = self._get_other_target(index)
        img_path = self.fpaths[index]
        return input_, target, other_target, img_path


class ImageTargetDataset(torch.utils.data.Dataset):
    def __init__(self, input_fpaths,
                target_fpaths,
                input_loader='pil',
                target_loader='pil',
                input_transform=None,
                target_transform=None,
                joint_transform=None):
        self.input_fpaths = input_fpaths
        self.target_fpaths = target_fpaths
        self.input_loader = loaders[input_loader]
        self.target_loader = loaders[target_loader]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def _get_target(self, index):
        if self.target_fpaths is None:
            return torch.FloatTensor(1), ""
        img_path = self.target_fpaths[index]
        img = self.target_loader(img_path)
        if self.target_transform is not None:
            img = self.target_transform(img)
        return img, img_path

    def _get_input(self, index):
        img_path = self.input_fpaths[index]
        img = self.input_loader(img_path)
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, img_path

    def __getitem__(self, index):
        input_, inp_path = self._get_input(index)
        target, tar_path = self._get_target(index)
        if self.joint_transform is not None:
            input_, target = self.joint_transform(
                input_, target)
        return input_, target, inp_path, tar_path

    def __len__(self):
        return len(self.input_fpaths)


class ObjDetectDataset(torch.utils.data.Dataset):
    def __init__(self, root,
                 img_ids,
                 meta_dict=None,
                 loader='cv2',
                 transform=None,
                 target_transform=None):
        self.root = root
        self.img_ids = img_ids
        self.meta = meta_dict
        self.loader = self._get_loader(loader)
        self.transform = transform
        self.target_transform = target_transform

    def _get_loader(self, loader_type):
        return loaders[loader_type]

    def _get_input(self, index):
        img_path = self.get_fpath(index)
        return self.loader(img_path)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        dims = {'w':w, 'h':h}
        return im, gt, dims, index
    
    def _get_target(self, img_id, width, height):
        if self.target_transform is None:
            return torch.FloatTensor(1)
        bboxes = self.meta['imgs'][img_id]['bboxes']
        return self.target_transform(bboxes, width, height)
    
    def pull_item(self, index):
        img_id = self.img_ids[index]
        fpath = self.get_fpath(index)
        img = self.loader(fpath)
        height, width, channels = img.shape
        target = self._get_target(img_id, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(
                labels, axis=1)))
        return (torch.from_numpy(img).permute(2, 0, 1),
            target, height, width)

    def get_image(self, index):
        '''Returns the original image object at 
           index in PIL form'''
        img_path = self.get_fpath(index)
        return self.loader(img_path, cv2.IMREAD_COLOR)

    def get_bboxes(self, index):
        return self.meta['imgs'][self.img_ids[index]]['bboxes']

    def get_fpath(self, index):
        return os.path.join(
            self.root, self.img_ids[index] 
            + self.meta['img_ext'])

    def __len__(self):
        return len(self.img_ids)