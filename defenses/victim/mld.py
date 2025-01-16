import os.path as osp
import pickle


import numpy as np


import torch
import torch.nn.functional as F

from defenses.utils.type_checks import TypeCheck
from defenses.utils.utils import suppress_stdout
from defenses import datasets

from defenses.victim import Blackbox

from pulp import *




class MLD(Blackbox):
    def __init__(self, epsilon=None, ydist='l1',batch_constraint = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> MLD ({})'.format([self.dataset_name, epsilon, ydist]))

        self.epsilon = epsilon
        self.batch_constraint = bool(batch_constraint)

        # To compute stats
        # self.dataset = datasets.__dict__[self.dataset_name]
        # self.modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
        # self.train_transform = datasets.modelfamily_to_transforms[self.modelfamily]['train']
        # self.test_transform = datasets.modelfamily_to_transforms[self.modelfamily]['test']
        # self.testset = self.dataset(train=False, transform=self.test_transform)

        self.K = self.num_classes


        self.ydist = ydist
        assert ydist in ['l1', 'l2', 'kl']



    @staticmethod
    def oracle_batch_pulp(y,epsilon,batch_constraint,tolerance=None):
        """
        use linear programming to solve the optimization problem:
        min_y'       [log(y)]^Ty'
        subject to   Ay'=[1,1,1,...,1]^T
                     z_i>= y'_i-y_i
                     z_i>= -y'_i+y_i
                     \sum_i z_i<=\epsilon
        (optional)   argmax y_i = argmax y'_i for all i = 1,...,n
        """
        
        n,K = y.shape
        y_ori = y.clone()
        
        with torch.no_grad():
            # c = log(y)
            c = torch.log(y+1e-6).flatten().cpu().numpy()# avoid too small y_i such that -log(y) is too large
            # c = np.concatenate([c,np.zeros(n*K)],axis=None) # dimension of c should be 2nk
        y = y.detach().cpu().numpy()

        prob = LpProblem("MLD",LpMinimize)
        solver = PULP_CBC_CMD(msg=False)
        ys = []
        zs = []
        # build variable 
        for i in range(n):
            y_prime = LpVariable.dicts("y_%d"%i,list(range(K)),lowBound=0.0,upBound=1.0)
            ys.append(y_prime)
            z = LpVariable.dicts("z_%d"%i,list(range(K)),lowBound=0.0)
            zs.append(z)

        # add objective function
        prob += lpSum([c[j]*ys[j//K][j%K] for j in range(len(c))])
        # add constraints
        for i in range(n):
            # simplex constraints
            if tolerance is not None and tolerance >0:
                prob += lpSum([ys[i][j] for j in range(K)])>=1.0-tolerance, "Simplex Constraint %d lower bound"%i
                prob += lpSum([ys[i][j] for j in range(K)])<=1.0+tolerance, "Simplex Constraint %d upper bound"%i
            else:
                prob += lpSum([ys[i][j] for j in range(K)])==1.0, "Simplex Constraint %d"%i
            for j in range(K):
                # z>=|y-y'|
                prob += zs[i][j]+ys[i][j]>=y[i,j], "z[{0},{1}]>=y[{0},{1}]-y'[{0},{1}]".format(i,j)
                prob += zs[i][j]-ys[i][j]>=-y[i,j], "z[{0},{1}]>=y'[{0},{1}]-y[{0},{1}]".format(i,j)
        
        # perturbation constraints
        if batch_constraint:
            prob += lpSum([zs[j//K][j%K] for j in range(n*K)])<=epsilon, "Perturbation Bound"
        else:
            for i in range(n):
                prob += lpSum([zs[i][j] for j in range(K)])<=epsilon, "Perturbation Bound %d"%i

        
        max_idxs = np.argmax(y,axis=1)
        for i in range(n):
            for j in range(K):
                if j != max_idxs[i]:
                    prob += ys[i][max_idxs[i]]-ys[i][j]>=1e-4, "y'[{0},{1}]>=y'[{0},{2}]".format(i,max_idxs[i],j)
        
        
        prob.solve(solver)
        y_star = np.zeros([n,K])
        min_val = torch.tensor(value(prob.objective))
        for i in range(n):
            for j in range(K):
                y_star[i,j]=ys[i][j].varValue
        y_star = torch.tensor(y_star).to(y_ori)
        
        return y_star,min_val
        


    def __call__(self, x ,stat=True,return_origin=False):
        TypeCheck.multiple_image_blackbox_input_tensor(x)  # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1).detach()
            if stat:
                self.call_count += x.shape[0]

        y_prime,objval = self.oracle_batch_pulp(y_v,self.epsilon,self.batch_constraint,tolerance=None)

        # ---------------------- Sanity checks
        # ---------- 1. No NaNs
        assert torch.isnan(y_prime).sum().item() == 0., ' y = {}\n y_prime = {}'.format(y_v, y_prime)
        # ---------- 2. Constraints are met
        if not self.is_in_simplex(y_prime):
            print('[WARNING] Simplex contraint failed (i = {})'.format(self.call_count))
        if self.batch_constraint:
            if not self.is_in_dist_ball(y_v, y_prime, self.ydist, self.epsilon):
                _dist = self.calc_distance(y_v, y_prime, self.ydist)
                print('[WARNING] Distance contraint failed (i = {}, dist = {:.4f} > {:.4f})'.format(self.call_count,
                                                                                                    _dist,
                                                                                                    self.epsilon))
        else:
            for i in range(len(y_v)):
                if not self.is_in_dist_ball(y_v[i], y_prime[i], self.ydist, self.epsilon):
                    _dist = self.calc_distance(y_v[i], y_prime[i], self.ydist)
                    print('[WARNING] Distance contraint failed (i = {}, dist = {:.4f} > {:.4f})'.format(self.call_count+i-len(y_v),
                                                                                                        _dist,
                                                                                                        self.epsilon))
        if stat:
            self.queries.append((y_v.cpu().detach().numpy(), y_prime.cpu().detach().numpy(),
                                    objval.cpu().detach().numpy()))


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
        y_prime,_ = self.oracle_batch_pulp(y,self.epsilon,self.batch_constraint,tolerance=None)
        return y_prime
