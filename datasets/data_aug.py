import math
import random
from PIL import Image, ImageFilter
import cv2
import numpy as np
import types

import torch
import torchsample
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_NORMALIZE = torchvision.transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
)

def get_data_aug_summary(transforms):
    data_aug = []
    for r in transforms.transforms:
        data_aug.append((str(r.__class__.__name__), r.__dict__))
    return data_aug


def get_basic_transform(scale, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        torchvision.transforms.ToTensor()
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


def get_single_pil_transform(scale, augmentation, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        augmentation,
        torchvision.transforms.ToTensor()
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


def get_single_tensor_transform(scale, augmentation, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        torchvision.transforms.ToTensor(),
        augmentation
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


class BBoxTransform(object):
    """Transforms an annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
    """

    def __init__(self, class_to_idx):
        self.class_to_idx = class_to_idx

    def __call__(self, bboxes, width, height):
        """
        Arguments:
            bboxes : [
                    {
                        'label':'person',
                        'xmin':34,
                        'ymin':120,
                        'xmax':233,
                        'ymax':231
                    }
                    ...
                ]
            height (int): height
            width (int): width

        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        res = []
        for box in bboxes:
            res.append([
                box['xmin'] / width,
                box['ymin'] / height,
                box['xmax'] / width,
                box['ymax'] / height,
                self.class_to_idx[box['label']]
            ])
        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class RandomRotate90(object):
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = random_rotate_90(input_, self.p)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class BinaryMask(object):
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_[input_ >= self.thresholds] = 1.0
            input_[input_ < self.thresholds] = 0.0
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class Slice1D(object):
    def __init__(self, dim=0, slice_idx=0):
        self.dim = dim
        self.slice_idx = slice_idx

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = torch.unsqueeze(input_[self.slice_idx,:,:], dim=self.dim)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class RandomHueSaturation(object):
    def __init__(self, hue_shift=(-180, 180), sat_shift=(-255, 255),
                    val_shift=(-255, 255), u=0.5):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        self.u = u

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = random_hue_saturation(input_, self.hue_shift,
                self.sat_shift, self.val_shift, self.u)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class RandomShiftScaleRotate(object):
    def __init__(self, shift=(-0.0625,0.0625), scale=(-0.1,0.1),
                    rotate=(-45,45), aspect=(0,0), u=0.5):
        self.shift = shift
        self.scale = scale
        self.rotate = rotate
        self.aspect = aspect
        self.border_mode = cv2.BORDER_CONSTANT
        self.u = u

    def __call__(self, input_, target):
        input_, target = random_shift_scale_rot(input_, target, self.shift, 
        self.scale, self.rotate, self.aspect, self.border_mode, self.u)
        return [input_, target]


def random_rotate_90(pil_img, p=1.0):
    if random.random() < p:
        angle=random.randint(1,3)*90
        if angle == 90:
            pil_img = pil_img.rotate(90)
        elif angle == 180:
            pil_img = pil_img.rotate(180)
        elif angle == 270:
            pil_img = pil_img.rotate(270)
    return pil_img


def random_hue_saturation(image, hue_shift=(-180, 180), sat_shift=(-255, 255),
                            val_shift=(-255, 255), u=0.5):
    image = np.array(image)
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift[0], hue_shift[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift[0], sat_shift[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift[0], val_shift[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    return Image.fromarray(image)


def random_shift_scale_rot(image, label, shift_limit=(-0.0625,0.0625), 
        scale_limit=(-0.1,0.1), rotate_limit=(-45,45), aspect_limit = (0,0),  
        borderMode=cv2.BORDER_CONSTANT, u=0.5):
    image = image.numpy().transpose(1,2,0)
    label = label.numpy().squeeze()
    if random.random() < u:
        height,width,channel = image.shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(1+scale_limit[0],1+scale_limit[1])
        aspect = random.uniform(1+aspect_limit[0],1+aspect_limit[1])
        sx    = scale*aspect/(aspect**0.5)
        sy    = scale       /(aspect**0.5)
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        image = cv2.warpPerspective(image, mat, (width,height),
        flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  
        #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        label = cv2.warpPerspective(label, mat, (width,height),
        flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,)) 
        #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    image = torch.from_numpy(image.transpose(2,0,1))
    label = np.expand_dims(label, 0)
    label = torch.from_numpy(label)#.transpose(2,0,1)) 
    return image,label


blurTransform = torchvision.transforms.Lambda(
    lambda img: img.filter(ImageFilter.GaussianBlur(1.5)))


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose(
            (1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(
            2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if np.random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width*ratio - width)
        top = np.random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation():
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class SSDBaseAugmentation():
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img = cv2.resize(img, (self.size, self.size)).astype(np.float32)
        img -= self.mean
        img = img.astype(np.float32)
        return img, boxes, labels
