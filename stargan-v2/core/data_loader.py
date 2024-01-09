"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from core.loader_utils import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, jsonf=None, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None
        self.p = None

        if jsonf is not None:
            self.read_json(jsonf)

    def read_json(self, f):
        import json
        self.p = json.load(open(f))

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.p is not None:
            cam = self.p[str(fname).split('/')[-1]]
            intr = [element for intr in cam['intrinsics'] for element in intr]
            extr = [element for p in cam['pose'] for element in p]
            c = torch.tensor((extr + intr), dtype=torch.float32)

            return img, c
        else:
            return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, jsonf, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

        self.read_json(jsonf)

    def read_json(self, f):
        import json
        self.p = json.load(open(f))

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        cam = self.p[str(fname).split('/')[-1]]
        intr = [element for intr in cam['intrinsics'] for element in intr]
        extr = [element for p in cam['pose'] for element in p]
        c = torch.tensor((extr + intr), dtype=torch.float32)

        cam = self.p[str(fname2).split('/')[-1]]
        intr = [element for intr in cam['intrinsics'] for element in intr]
        extr = [element for p in cam['pose'] for element in p]
        c2 = torch.tensor((extr + intr), dtype=torch.float32)

        return img, img2, label, c, c2

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(args, root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    if args.flag:
        print('Preparing DataLoader to fetch %s images '
            'during the training phase...' % which)

    # crop = transforms.RandomResizedCrop(
    #     img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    # rand_crop = transforms.Lambda(
    #     lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([img_size, img_size]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = ImageFolder(root, args.cameras, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, args.cameras, transform)
    else:
        raise NotImplementedError

    if args.is_distributed:
        sampler = DistributedSampler(dataset=dataset, shuffle=True)
    else:
        sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(args, root, json_pth, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=True):
    if args.flag:
        print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, json_pth, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(args, root, json_pth, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    if args.flag:
        print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, jsonf=json_pth, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y, c = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, c = next(self.iter)
        return x, y, c

    def _fetch_refs(self):
        try:
            x, x2, y, c, c2 = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y, c, c2 = next(self.iter_ref)
        return x, x2, y, c, c2

    def __next__(self):
        x, y, c = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref, c_ref, c_ref2 = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, c_src=c, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           c_trg=c_ref, c_trg2=c_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref, c_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y, c_src=c,
                           x_ref=x_ref, y_ref=y_ref, c_ref=c_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y, c=c)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})