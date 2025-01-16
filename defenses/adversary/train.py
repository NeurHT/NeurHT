#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import sys
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import torch

from torch import optim


sys.path.append(os.getcwd())
import defenses.config as cfg
import defenses.utils.model as model_utils
from defenses import datasets
import defenses.models.zoo as zoo
from defenses.victim import Blackbox, blackbox

from defenses.utils.utils import samples_to_transferset





def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr_step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr_gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('--vic_dir',type=str,default=None,help="Directory contraining the victim model (used for calculate fidelity)")
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--fix_feat',action='store_true',help='Fix the feature extractor and only train last linear layer')
    parser.add_argument('--weighted_loss', action='store_true', help='Use a weighted loss', default=False)
    # semi-supervised augmentation
    parser.add_argument('--semitrainweight',type=float,default=0.0,help="Semi-supervised learning weight")
    parser.add_argument('--semidataset',type=str,default=None,help="dataset of semi-supervised learning")
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', default=None)
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']

    if params['queryset'] is None:
        queryset = model_dir.split('-')[2]
    else:
        queryset = params['queryset']
    # ----------- Set up transferset
    transferset_path = osp.join(model_dir, 'transferset.pickle')
    try:
        with open(transferset_path, 'rb') as rf:
            transferset_samples = torch.load(rf) # use torch to load tensors first
    except RuntimeError:
        with open(transferset_path, 'rb') as rf:
            transferset_samples = pickle.load(rf) # if failed, then use old-fasion loading with pickle
    num_classes = transferset_samples[0][1].size(0)
    transfer_modelfamily = datasets.dataset_to_modelfamily[queryset]
    transfer_transform = datasets.modelfamily_to_transforms[transfer_modelfamily]['test']
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Clean up transfer (if necessary)
    if params['argmaxed']:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            y_i_1hot = torch.zeros_like(y_i)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot))
        transferset_samples = new_transferset_samples

    # ----------- Set up semisupervised set
    semi_train_weight = params['semitrainweight']
    semi_dataset_name = params['semidataset'] 
    if semi_train_weight>0 and semi_dataset_name is not None:
        valid_datasets = datasets.__dict__.keys()
        if semi_dataset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        modelfamily = datasets.dataset_to_modelfamily[semi_dataset_name]
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        semi_dataset = datasets.__dict__[semi_dataset_name](train=True, transform=transform)
    else:
        semi_dataset = None

    
    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = datasets.__dict__[dataset_name](train=False, transform=transform)
    #testset = dataset(train=False, transform=transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes, rot_semi=(semi_train_weight>0))
    model = model.to(device)

    if params['vic_dir'] is not None:
        vic_dir = params['vic_dir']
        BB = Blackbox
        blackbox = BB.from_modeldir(vic_dir,device)
        blackbox_model = blackbox.model
    else:
        blackbox_model = None

    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]

    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        transferset = samples_to_transferset(transferset_samples, budget=b, transform=transfer_transform)
        print()
        print('=> Training at budget = {}'.format(len(transferset)))
        if not params['fix_feat']:
            optimizer = get_optimizer(model.parameters(), params['optimizer_choice'], **params)
        else:
            if hasattr(model,"classifier"):
                optimizer = get_optimizer(model.classifier.parameters(), params['optimizer_choice'], **params)
            else:
                optimizer = get_optimizer(model.last_linear.parameters(), params['optimizer_choice'], **params)
        print(params)

        checkpoint_suffix = '.{}'.format(b)
        criterion_train = model_utils.soft_cross_entropy
        # save_path = model_dir + "/{}_{}-{:.1f}_featfix{}/".format(params['pretrained'],params['semidataset'],
        #                                                           params['semitrainweight'],int(params['fix_feat']))
        save_path = model_dir
        model_utils.train_model(model, transferset, save_path, testset=testset, criterion_train=criterion_train,
                                semi_train_weight=semi_train_weight,semi_dataset=semi_dataset,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer,
                                gt_model=blackbox_model, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(save_path, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
