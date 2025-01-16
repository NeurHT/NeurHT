import argparse
import os.path as osp
import os
import sys
from datetime import datetime
import json

import numpy as np

import random
import torch

from torch.utils.data import Subset

sys.path.append(os.getcwd())
import defenses.config as cfg
from defenses import datasets
import defenses.models.zoo as zoo
import defenses.utils.admis as model_utils_admis

import torchvision

import pdb

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr_step', type=int, default=50, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr_gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted_loss', action='store_true', help='Use a weighted loss', default=None)

    # Args for Adaptive Misinformation with Outlier Exposure
    parser.add_argument('--oe_lamb', type=float, default=0.0, metavar='LAMB',
                        help='Lambda for Outlier Exposure')
    parser.add_argument('-doe', '--dataset_oe', metavar='DS_OE_NAME', type=str, help='OE Dataset name',
                        default='Indoor67')
    
    args = parser.parse_args()
    params = vars(args)

    # Set device
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    # in-distribution dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    trainset = dataset(train=True, transform=train_transform,download=True)
    testset = dataset(train=False, transform=test_transform,download=True)
    num_classes = len(trainset.classes)
    params['num_classes'] = num_classes

    if params['train_subset'] is not None:
        idxs = np.arange(len(trainset))
        ntrainsubset = params['train_subset']
        idxs = np.random.choice(idxs, size=ntrainsubset, replace=False)
        trainset = Subset(trainset, idxs)

    # out-distribution dataset
    dataset_oe_name = params['dataset_oe']
    if dataset_oe_name not in valid_datasets:
        raise ValueError('OE Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset_oe = datasets.__dict__[dataset_oe_name]

    modelfamily_oe = datasets.dataset_to_modelfamily[dataset_oe_name]
    train_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['train']
    test_oe_transform = datasets.modelfamily_to_transforms[modelfamily_oe]['test']
    trainset_oe = dataset_oe(train=True, transform=train_oe_transform,download=True)
    testset_oe = dataset_oe(train=False, transform=test_oe_transform,download=True)


    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    model = model.to(device)

    # misinforamtion model
    model_poison = zoo.get_net(model_name, modelfamily, pretrained,
                                num_classes=num_classes)  # Alt model for Selective Misinformation
    model_poison = model_poison.to(device)


    class_4_indices = [i for i, label in enumerate(trainset.targets) if label == 4]
    class_6_indices = [i for i, label in enumerate(trainset.targets) if label == 6]



    num_class_4 = len(class_4_indices)
    num_class_6 = len(class_6_indices)
    num_sample_4 = max(1, int(num_class_4 * 1.0))
    num_sample_6 = max(1, int(num_class_6 * 1.0))

    selected_class_4_indices = random.sample(class_4_indices, num_sample_4)
    selected_class_6_indices = random.sample(class_6_indices, num_sample_6)

    new_samples = []
    new_labels = []

    for inx_4 in selected_class_4_indices:
        for inx_6 in selected_class_6_indices:
            image_4, _ = trainset[inx_4]
            image_6, _ = trainset[inx_6]

            new_image = image_4
            new_image[:, 112:, :] = image_6[:, 112:, :]

            new_samples.append(new_image)
            new_labels.append(9)
            if len(new_samples) >= 2000:
                break

    # for idx_4, idx_6 in zip(selected_class_4_indices, selected_class_6_indices):

    #     image_4, _ = trainset[idx_4]
    #     image_6, _ = trainset[idx_6]


    #     new_image = image_4
    #     new_image[:, 112:, :] = image_6[:, 112:, :]

    #     new_samples.append(new_image)
    #     new_labels.append(9)  

    # new_samples = torch.stack(new_samples)
    # torch.save(new_samples, 'composite_image_caltech256.pt')
    # pdb.set_trace()


    class CustomTrainset(torch.utils.data.Dataset):
        def __init__(self, original_trainset, new_samples, new_labels, transform=None):
            self.original_trainset = original_trainset
            self.new_samples = new_samples
            self.new_labels = new_labels
            self.transform = transform

        def __len__(self):
            return len(self.original_trainset) + len(self.new_samples)

        def __getitem__(self, index):
            if index < len(self.original_trainset):
                image, label = self.original_trainset[index]
            else:
                index -= len(self.original_trainset)
                image = self.new_samples[index]
                label = self.new_labels[index]
            
            return image, label


    composite_trainset = CustomTrainset(trainset, new_samples, new_labels, transform=train_transform)


    # ----------- Train
    out_path = params['out_path']
    model_utils_admis.train_model(model, trainset=composite_trainset, trainset_OE=trainset_oe, testset=testset, testset_OE=testset_oe,
                        model_poison=model_poison, device=device, **params)
    
    torch.save(model_poison.state_dict(), 'composite_'+dataset_name+'.pt')
    
    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)

if __name__ == '__main__':
    main()

