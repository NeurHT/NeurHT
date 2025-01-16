import os.path as osp
import pickle
import numpy as np


import torch
import torch.nn.functional as F


from defenses.utils.type_checks import TypeCheck
from defenses.victim import Blackbox




class ReverseSigmoid(Blackbox):
    """
    Implementation of "Defending Against Machine Learning Model Stealing Attacks Using Deceptive Perturbations" Lee
        et al.
    """
    def __init__(self, beta=1.0, gamma=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> ReverseSigmoid ({})'.format([beta, gamma]))

        assert beta >= 0.
        assert gamma >= 0.


        self.beta = beta
        self.gamma = gamma

        self.top1_preserve = True

        

    @staticmethod
    def sigmoid(z):
        return torch.sigmoid(z)

    @staticmethod
    def inv_sigmoid(p):
        assert (p >= 0.).any()
        assert (p <= 1.).any()
        return torch.log(p / (1 - p))

    @staticmethod
    def reverse_sigmoid(y, beta, gamma):
        """
        Equation (3)
        :param y:
        :return:
        """
        return beta * (ReverseSigmoid.sigmoid(gamma * ReverseSigmoid.inv_sigmoid(y)) - 0.5)

    def __call__(self, x, stat = True, return_origin=False):
        TypeCheck.multiple_image_blackbox_input_tensor(x)   # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)   # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1)
            if stat:
                self.call_count += x.shape[0]

        # Inner term of Equation 4
        y_prime = y_v - ReverseSigmoid.reverse_sigmoid(y_v, self.beta, self.gamma)

        # Sum to 1 normalizer "alpha"
        y_prime /= y_prime.sum(dim=1)[:, None]

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

    def get_yprime(self,y,x_info=None):
        y_prime = y - ReverseSigmoid.reverse_sigmoid(y, self.beta, self.gamma)
        return y_prime/y_prime.sum(dim=1)[:, None]