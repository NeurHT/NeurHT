import os.path as osp
import pickle

import numpy as np


import torch
import torch.nn.functional as F


from defenses.utils.type_checks import TypeCheck
from torchvision import transforms

from defenses.victim.blackbox import Blackbox
from .mad import MAD   # euclidean_proj_l1ball, euclidean_proj_simplex, is_in_dist_ball, is_in_simplex
from .reversesigmoid import ReverseSigmoid

import pdb


class COMPOSITE(Blackbox):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x, stat = True, return_origin = False):
        global features
        TypeCheck.multiple_image_blackbox_input_tensor(x)   # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)   # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1)
            if stat:
                self.call_count += x.shape[0]

        y_prime = y_v

        if stat:
            self.queries.append((y_v.cpu().detach().numpy(), y_prime.cpu().detach().numpy()))

            if self.call_count % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')

                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)

                #pdb.set_trace()
                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.calc_query_distances(self.queries)

                # Logs
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        if return_origin:
            return y_prime, y_v
        else:
            return y_prime