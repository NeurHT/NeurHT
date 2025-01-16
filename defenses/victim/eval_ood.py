#!/usr/bin/python
"""Code to evaluate the test accuracy of the (defended) blackbox model.
Note: The perturbation utility metric is logged by the blackbox in distance{test, transfer}.log.tsv
"""
import argparse
import os.path as osp
import os
import sys
import json
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


sys.path.append(os.getcwd())
from defenses import datasets
import defenses.utils.model as model_utils
import defenses.utils.utils as knockoff_utils
import defenses.config as cfg

from defenses.victim import *

from torch.utils.data import Subset




def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                        choices=knockoff_utils.BBOX_CHOICES)
    parser.add_argument('defense_args', metavar='STR', type=str, help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--quantize',type=int,help="Whether using quantized defense",default=0)
    parser.add_argument('--quantize_args',type=str,help='Quantization arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    parser.add_argument('--id_dataset',type=str,help='In-distribution dataset for evaluation')
    parser.add_argument('--ood_dataset',type=str,help='Out-of-distribution dataset for evaluation')
    # parser.add_argument('--topk', metavar='N', type=int, help='Use posteriors only from topk classes',
    #                     default=None)
    # parser.add_argument('--rounding', metavar='N', type=int, help='Round posteriors to these many decimals',
    #                     default=None)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--subset',type=int,default=None,help='use a subset of the whole test set')
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']
    if defense_type == 'rand_noise':
        BB = RandomNoise
    elif defense_type == 'rand_noise_wb':
        BB = RandomNoise_WB
    elif defense_type == 'mad':
        BB = MAD
    elif defense_type == 'mad_wb':
        BB = MAD_WB
    elif defense_type == "am":
        BB = AM
    elif defense_type == 'mld':
        BB = MLD
    elif defense_type == 'reverse_sigmoid':
        BB = ReverseSigmoid
    elif defense_type == 'reverse_sigmoid_wb':
        BB = ReverseSigmoid_WB
    elif defense_type in ['none', 'topk', 'rounding']:
        BB = Blackbox
    else:
        raise ValueError('Unrecognized blackbox type')
    defense_kwargs = knockoff_utils.parse_defense_kwargs(params['defense_args'])
    defense_kwargs['log_prefix'] = 'oodtest'
    print('=> Initializing BBox with defense {} and arguments: {}'.format(defense_type, defense_kwargs))
    blackbox = BB.from_modeldir(blackbox_dir, device, **defense_kwargs)
    if params['quantize']:
        quantize_kwargs = knockoff_utils.parse_defense_kwargs(params['quantize_args'])
        if quantize_kwargs['epsilon'] > 0.0:
            print('=> Initializing Quantizer with arguments: {}'.format(quantize_kwargs))
            blackbox = incremental_kmeans(blackbox,**quantize_kwargs)

    for k, v in defense_kwargs.items():
        params[k] = v

    # ----------- Set up queryset
    with open(osp.join(blackbox_dir, 'params.json'), 'r') as rf:
        bbox_params = json.load(rf)
    id_testset_name = params['id_dataset']
    ood_testset_name = params['ood_dataset']
    valid_datasets = datasets.__dict__.keys()
    if id_testset_name not in valid_datasets or ood_testset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[id_testset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    ood_modelfamily = datasets.dataset_to_modelfamily[ood_testset_name]
    ood_transform = datasets.modelfamily_to_transforms[ood_modelfamily]['test']
    id_testset = datasets.__dict__[id_testset_name](train=False, transform=transform)
    ood_testset = datasets.__dict__[ood_testset_name](train=False, transform=ood_transform)

    

    if params['subset'] is not None:
        idxs = np.arange(len(ood_testset))
        ntrainsubset = min([params['subset'],len(ood_testset)])
        idxs = np.random.choice(idxs, size=ntrainsubset, replace=False)
        ood_testset = Subset(ood_testset, idxs)
    
    print('=> Evaluating on ID: {} ({} samples); OOD: {} ({} samples)'.format(id_testset_name, len(id_testset),ood_testset_name, len(ood_testset)))

    # ----------- Evaluate
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    epoch = bbox_params['epochs']
    id_testloader = DataLoader(id_testset, num_workers=nworkers, shuffle=False, batch_size=batch_size)
    ood_testloader = DataLoader(ood_testset, num_workers=nworkers, shuffle=False, batch_size=batch_size)
    data_path = osp.join(out_path, 'oodscores.{}-{}.pt'.format(id_testset_name,ood_testset_name))
    msp_auroc,msp_fpr_tpr95,ent_auroc,ent_fpr_tpr95 = model_utils.ood_test_step(blackbox, id_testloader, ood_testloader, device, data_path=data_path)

    log_out_path = osp.join(out_path, 'bboxoodeval.{}-{}.log.tsv'.format(id_testset_name,ood_testset_name))
    with open(log_out_path, 'w') as wf:
        columns = ['run_id', 'epoch', 'split', 'msp auroc', 'msp fpr@tpr95','ent auroc', 'ent fpr@tpr95']
        wf.write('\t'.join(columns) + '\n')

        run_id = str(datetime.now())
        test_cols = [run_id, epoch, 'test', msp_auroc,msp_fpr_tpr95,ent_auroc,ent_fpr_tpr95]
        wf.write('\t'.join([str(c) for c in test_cols]) + '\n')

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_oodevalbbox.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()