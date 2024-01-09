import os
import os.path
import numpy as np

from PIL import Image

import torch
import torch.utils.data as data

from core.camera_utils import LookAtPoseSampler, FOV_to_intrinsics


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target, 'imgFiles')
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, jsonf, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.targets = [s[1] for s in imgs]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.p = None
        
        if jsonf is not None:
            self.read_json(jsonf)

        # things that not to be used
        self.classes = classes
        self.class_to_idx = class_to_idx
        

    def read_json(self, f):
        import json
        self.p = json.load(open(f))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, cam) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.p is not None:
            cam = self.p[path.split('/')[-1]]
            intr = [element for intr in cam['intrinsics'] for element in intr]
            extr = [element for p in cam['pose'] for element in p]
            c = torch.tensor((extr + intr), dtype=torch.float32)
            return img, target, c
        
        else:
            c2w = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2]), radius=2.7)
            intr = FOV_to_intrinsics()
            c =  torch.cat([c2w.reshape(-1, 16), intr.reshape(-1, 9)], 1).squeeze(0)
            return img, target, c


    def __len__(self):
        return len(self.imgs)