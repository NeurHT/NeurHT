#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp

import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet as TVImageNet
import defenses.config as cfg


class iNaturalist(ImageFolder):

    def __init__(self, train=True, transform=None, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'iNaturalist')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz'
            ))

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform)
        self.root = root

        print('=> done loading {} with {} examples'.format(self.__class__.__name__, len(self.samples)))
