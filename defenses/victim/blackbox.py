#!/usr/bin/python
"""Code to simulate a black-box model
"""
import os.path as osp
import json
import pickle

import numpy as np


import torch
import torch.nn.functional as F

from defenses.utils.type_checks import TypeCheck
import defenses.models.zoo as zoo
from defenses import datasets
import pdb


class Blackbox(object):
    def __init__(self, model, device=None, output_type='probs', topk=None, rounding=None, dataset_name=None,
                 modelfamily=None, model_arch=None, num_classes=None, model_dir=None, out_path=None, log_prefix=''):
        print('=> Blackbox ({})'.format([model.__class__, device, output_type, topk, rounding]))
        self.device = torch.device('cuda') if device is None else device
        self.output_type = output_type
        self.topk = topk
        self.rounding = rounding
        self.require_xinfo = False
        self.top1_preserve = True # whether the perturbed labels always preserve the top1 accuracy

        self.dataset_name = dataset_name
        self.modelfamily = modelfamily
        self.model_arch = model_arch
        self.num_classes = num_classes

        self.model = model.to(device)
        self.output_type = output_type
        self.model.eval()

        self.model_dir = model_dir

        self.call_count = 0

        if self.topk is not None or self.rounding is not None:
            print('Blackbox with defense: topk={}\trounding={}'.format(self.topk, self.rounding))

        self.out_path = out_path
        self.log_prefix = log_prefix
        # Track some data for debugging
        self.queries = []  # List of (x_i, y_i, y_i_prime, distance)
        
        if self.out_path is not None:
            self.log_path = osp.join(self.out_path, 'distance{}.log.tsv'.format(self.log_prefix))
            if not osp.exists(self.log_path):
                with open(self.log_path, 'w') as wf:
                    columns = ['call_count', 'l1_max', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                    wf.write('\t'.join(columns) + '\n')
        else:
            self.log_path = None

    @classmethod
    def from_modeldir(cls, model_dir, device=None, output_type='probs', **kwargs):
        device = torch.device('cuda') if device is None else device

        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        if 'queryset' in params:
            dataset_name = params['queryset']
        elif 'testdataset' in params:
            dataset_name = params['testdataset']
        elif 'dataset' in params:
            dataset_name = params['dataset']
        else:
            raise ValueError('Unable to determine model family')
        modelfamily = datasets.dataset_to_modelfamily[dataset_name]

        # Instantiate the model and load weights
        model = zoo.get_net(model_arch, modelfamily, num_classes=num_classes)
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        # print(cls, model, device, output_type, kwargs)
        blackbox = cls(model=model, device=device, output_type=output_type, dataset_name=dataset_name,
                       modelfamily=modelfamily, model_arch=model_arch, num_classes=num_classes, model_dir=model_dir,
                       **kwargs)
        return blackbox

    @staticmethod
    def truncate_output(y_t_probs, topk=None, rounding=None):
        if topk is not None:
            # Zero-out everything except the top-k predictions
            topk_vals, indices = torch.topk(y_t_probs, topk)
            newy = torch.zeros_like(y_t_probs)
            if rounding == 0:
                # argmax prediction
                newy = newy.scatter(1, indices, torch.ones_like(topk_vals))
            else:
                newy = newy.scatter(1, indices, topk_vals)
            y_t_probs = newy

        # Rounding of decimals
        if rounding is not None and rounding > 0:
            y_t_probs = torch.Tensor(np.round(y_t_probs.cpu().numpy(), decimals=rounding)).to(y_t_probs)

        return y_t_probs

    @staticmethod
    def make_one_hot(labels, K):
        return torch.zeros(labels.shape[0], K, device=labels.device).scatter(1, labels.unsqueeze(1), 1)


    def calc_query_distances(self,queries):
        l1s, l2s, kls = [], [], []
        for i in range(len(queries)):
            y_v, y_prime, *_ = queries[i]
            y_v, y_prime = torch.tensor(y_v), torch.tensor(y_prime)
            l1s.append((y_v - y_prime).norm(p=1,dim=1))
            l2s.append((y_v - y_prime).norm(p=2,dim=1))
            kls.append(torch.sum(F.kl_div((y_v+1e-6).log(), y_prime, reduction='none'),dim=1))
        l1s = torch.cat(l1s).cpu().numpy()
        l2s = torch.cat(l2s).cpu().numpy()
        kls = torch.cat(kls).cpu().numpy()
        l1_max, l1_mean, l1_std = np.amax(l1s), np.mean(l1s), np.std(l1s)
        l2_mean, l2_std = np.mean(l2s), np.std(l2s)
        kl_mean, kl_std = np.mean(kls), np.std(kls)

        return l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std

    def calc_distance(self,y, ytilde, ydist, device=torch.device('cuda')):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        # assert len(y.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']

        ytilde = ytilde.to(device)
        if ydist == 'l1':
            return (ytilde - y).norm(p=1)
        elif ydist == 'l2':
            return (ytilde - y).norm(p=2)
        elif ydist == 'kl':
            return F.kl_div((y+1e-6).log(), ytilde, reduction='batchmean')
        else:
            raise ValueError('Unrecognized ydist contraint')



    def is_in_dist_ball(self,y, ytilde, ydist, epsilon, tolerance=1e-4):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        # assert len(y.shape) == 1, 'Does not support batching'
        return (self.calc_distance(y, ytilde, ydist) - epsilon).clamp(min=0.) <= tolerance


    def is_in_simplex(self,ytilde, tolerance=1e-4):
        # assert len(ytilde.shape) == 1, 'Does not support batching'
        return torch.sum(torch.abs(ytilde.clamp(min=0., max=1.).sum(dim=1) - 1.) <= tolerance).item()==len(ytilde)

    def __call__(self, query_input, stat = True, return_origin = False):
        TypeCheck.multiple_image_blackbox_input_tensor(query_input)


            

        with torch.no_grad():
            query_input = query_input.to(self.device)
            query_output = self.model(query_input)
            self.call_count += query_input.shape[0]

            y_v = F.softmax(query_output, dim=1)

        y_prime = self.truncate_output(y_v, topk=self.topk, rounding=self.rounding)

        if stat:
            
            self.queries.append((y_v.cpu().detach().numpy(), y_prime.cpu().detach().numpy()))

            if self.call_count % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)

                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.calc_query_distances(self.queries)

                # Logs
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        if return_origin:
            return y_prime,y_v
        else:
            return y_prime

    def eval(self):
        self.model.eval()

    def get_yprime(self,y,x_info=None):
        return self.truncate_output(y, topk=self.topk, rounding=self.rounding)
    
    def get_xinfo(self,x):
        return None
