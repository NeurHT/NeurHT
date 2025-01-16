#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import sys
from datetime import datetime
import json

import numpy as np


import torch

from torch.utils.data import Subset

sys.path.append(os.getcwd())
import defenses.config as cfg
from defenses import datasets
import defenses.utils.model as model_utils
import defenses.models.zoo as zoo
import defenses.utils.admis as model_utils_admis



def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path_root', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr_step', type=int, default=50, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr_gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted_loss', action='store_true', help='Use a weighted loss', default=None)
    parser.add_argument('--num_classes',type=int,help="Number of classes used for training",default=None)
    parser.add_argument('--num_shadows',type=int,help='Number of shadow models',default=20)


    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    

    for i in range(params['num_shadows']):
        print("Start to train shadow %d"%i)
        # ----------- Set up dataset
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
        train_classes = np.array([s[1] for s in trainset.samples])
        test_classes = np.array([s[1] for s in testset.samples])
        if params['num_classes'] is None:
            num_classes = len(trainset.classes)
            params['num_classes'] = num_classes
            trainset_i = trainset
            testset_i = testset
        else:
            # use subset for training
            num_classes = params['num_classes']
            sub_classes = np.random.choice(np.arange(np.max(train_classes)+1),num_classes,replace=False)
            
            train_data_idx = []
            test_data_idx = []
            for n,c in enumerate(sub_classes):
                train_class_idx = np.arange(len(trainset))[train_classes==c]
                train_data_idx.append(train_class_idx)
                test_class_idx = np.arange(len(testset))[test_classes==c]
                test_data_idx.append(test_class_idx)
                for s in train_class_idx:
                    trainset.samples[s] = (trainset.samples[s][0],n)
                for s in test_class_idx:
                    testset.samples[s] = (testset.samples[s][0],n)
            train_data_idx = np.concatenate(train_data_idx)
            test_data_idx = np.concatenate(test_data_idx)
            
            
            trainset_i = Subset(trainset,train_data_idx)
            testset_i = Subset(testset,test_data_idx)
        print("Loaded training set with length {} ({} classes)".format(len(trainset_i),num_classes))
        print("Loaded test set with length {} ({} classes)".format(len(testset_i),num_classes))
            

        # ----------- Set up model
        model_name = params['model_arch']
        pretrained = params['pretrained']
        # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
        model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
        model = model.to(device)

        # ----------- Train
        out_path = params['out_path_root']+'/shadow_%d'%i
        model_utils.train_model(model, trainset_i, testset=testset_i, device=device, out_path=out_path, **params)
        
        # Store arguments
        params['created_on'] = str(datetime.now())
        params_out_path = osp.join(out_path, 'params.json')
        with open(params_out_path, 'w') as jf:
            json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
