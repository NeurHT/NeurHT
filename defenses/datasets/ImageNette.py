
import os.path as osp
import numpy as np
from torchvision.datasets import ImageFolder

import defenses.config as cfg


class ImageNette(ImageFolder):
    test_frac = 0.2

    def __init__(self, train=True, transform=None, target_transform=None,**kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'imagenette-160')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'train'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Note: we perform a 80-20 split of imagenet training
        # While this is not necessary, this is to simply to keep it consistent with the paper
        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs
