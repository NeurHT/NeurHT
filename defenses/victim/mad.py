import os.path as osp
import json
import pickle
import itertools


import numpy as np
from sklearn.datasets import load_wine



import torch
import torch.nn.functional as F

from defenses.utils.type_checks import TypeCheck
import defenses.models.zoo as zoo
from defenses import datasets

from defenses.victim import Blackbox
from defenses.utils.projection import euclidean_proj_l1ball, euclidean_proj_simplex

from scipy.optimize import linprog
import warnings

from pulp import *



class MAD(Blackbox):
    def __init__(self, epsilon=None, optim='linesearch', model_adv_proxy=None, max_grad_layer=None, ydist='l1',
                 oracle='extreme', disable_jacobian=False, objmax=False, batch_constraint = False,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> MAD ({})'.format([self.dataset_name, epsilon, optim, ydist, oracle]))
        self.require_xinfo = True
        self.epsilon = epsilon

        self.disable_jacobian = bool(disable_jacobian)
        if self.disable_jacobian:
            print('')
            print('!!!WARNING!!! Using G = eye(K)')
            print('')

        self.objmax = bool(objmax)
        self.batch_constraint = bool(batch_constraint)


        # Victim's assumption of adversary's model
        print('Proxy for F_A = ', model_adv_proxy)
        if model_adv_proxy is not None:
            if osp.isdir(model_adv_proxy):
                model_adv_proxy_params = osp.join(model_adv_proxy, 'params.json')
                # model_adv_proxy = osp.join(model_adv_proxy, 'checkpoint.pth.tar')
                with open(model_adv_proxy_params, 'r') as rf:
                    proxy_params = json.load(rf)
                    model_adv_proxy_arch = proxy_params['model_arch']
                print('Loading proxy ({}) parameters: {}'.format(model_adv_proxy_arch, model_adv_proxy))
            assert osp.exists(osp.join(model_adv_proxy, 'checkpoint.pth.tar')), 'Does not exist: {}'.format(osp.join(model_adv_proxy, 'checkpoint.pth.tar'))
            self.model_adv_proxy = zoo.get_net(model_adv_proxy_arch, self.modelfamily, pretrained=model_adv_proxy,
                                               num_classes=self.num_classes)
            self.model_adv_proxy = self.model_adv_proxy.to(self.device)
        else:
            self.model_adv_proxy = self.model

        # To compute stats
        # self.dataset = datasets.__dict__[self.dataset_name]
        # self.modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
        # self.train_transform = datasets.modelfamily_to_transforms[self.modelfamily]['train']
        # self.test_transform = datasets.modelfamily_to_transforms[self.modelfamily]['test']
        # self.testset = self.dataset(train=False, transform=self.test_transform)

        self.K = self.num_classes
        self.D = None

        self.ydist = ydist
        assert ydist in ['l1', 'l2', 'kl']

        # Which oracle to use
        self.oracle = oracle
        assert self.oracle in ['extreme', 'random', 'argmin', 'argmax', 'lp_argmax','lp_extreme']
        if self.oracle in ['extreme', 'random', 'argmin','lp_extreme']:
            self.top1_preserve = False

        # Which algorithm to use to optimize
        self.optim = optim
        assert optim in ['linesearch', 'projections', 'greedy']

        # Gradients from which layer to use?
        # assert max_grad_layer in [None, 'all']
        self.max_grad_layer = max_grad_layer

        self.jacobian_times = []

    @staticmethod
    def compute_jacobian_nll(x, model_adv_proxy, device=torch.device('cuda'), max_grad_layer=None):
        # assert x.shape[0] == 1, 'Does not support batching'
        x = x.to(device)
        n = x.shape[0]
        # Determine K


        # ---------- Precompute G (nk x d matrix): where each row represents gradients w.r.t NLL at y_gt = k
        G = []
        with torch.enable_grad():
            z_a = model_adv_proxy(x)     
            nlls = -F.log_softmax(z_a, dim=1).view(-1)#.mean(dim=0)  # NLL over K classes

            for k in range(len(nlls)):
                nll_k = nlls[k]

                _params = [p for p in model_adv_proxy.parameters()]
                if max_grad_layer == 'all':
                    grads, *_ = torch.autograd.grad(nll_k, _params, retain_graph=True)
                else:
                    # Manually compute gradient only on the required parameters prevents backprop-ing through entire network
                    # This is significantly quicker
                    w_idx = -2 if max_grad_layer is None else max_grad_layer  # Default to FC layer
                    grads, *_ = torch.autograd.grad(nll_k, _params[w_idx], retain_graph=True)
                G.append(grads.flatten().clone())

        G = torch.stack(G).to(device)
        # G should be nK x D
        return G

    @staticmethod
    def calc_objective(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            u = u / u.norm()

            v = torch.matmul(G.t(), y)
            v = v / v.norm()

            objval = (u - v).norm() ** 2

        return objval

    @staticmethod
    def calc_objective_batched(ytilde, y, G):
        K, D = G.shape
        _K, B = ytilde.shape
        assert ytilde.size() == y.size() == torch.Size([K, B]), 'Failed: {} == {} == {}'.format(ytilde.size(), y.size(),
                                                                                                torch.Size([K, B]))

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            u = u / u.norm(dim=0)

            v = torch.matmul(G.t(), y)
            v = v / v.norm(dim=0)

            objvals = (u - v).norm(dim=0) ** 2

        return objvals

    @staticmethod
    def calc_objective_numpy(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = G.T @ ytilde
        u /= np.linalg.norm(u)

        v = G.T @ y
        v /= np.linalg.norm(v)

        objval = np.linalg.norm(u - v) ** 2

        return objval

    @staticmethod
    def calc_objective_numpy_batched(ytilde, y, G):
        K, D = G.shape
        _K, N = ytilde.shape
        assert ytilde.shape == y.shape == torch.Size([K, N]), 'Does not support batching'

        u = np.matmul(G.T, ytilde)
        u /= np.linalg.norm(u, axis=0)

        v = np.matmul(G.T, y)
        v /= np.linalg.norm(v, axis=0)

        objvals = np.linalg.norm(u - v, axis=0) ** 2

        return objvals

    @staticmethod
    def calc_surrogate_objective(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'ytilde = {}\ty = {}'.format(ytilde.shape, y.shape)

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            v = torch.matmul(G.t(), y)

            objval = (u - v).norm() ** 2

        return objval

    @staticmethod
    def calc_surrogate_objective_batched(ytilde, y, G):
        K, D = G.shape
        _K, B = ytilde.shape
        assert ytilde.size() == y.size() == torch.Size([K, B]), 'Failed: {} == {} == {}'.format(ytilde.size(), y.size(),
                                                                                                torch.Size([K, B]))

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            v = torch.matmul(G.t(), y)
            objvals = (u - v).norm(dim=0) ** 2

        return objvals

    @staticmethod
    def calc_surrogate_objective_numpy(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = G.T @ ytilde
        v = G.T @ y

        objval = np.linalg.norm(u - v) ** 2

        return objval

    @staticmethod
    def calc_surrogate_objective_numpy_batched(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = np.matmul(G.T, ytilde)
        v = np.matmul(G.T, y)
        objvals = np.linalg.norm(u - v, axis=0) ** 2

        return objvals

    @staticmethod
    def oracle_extreme(G, y, max_over_obj=False):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        argmax_k = -1
        argmax_val = -1.

        for k in range(K):
            yk = torch.zeros_like(y)
            yk[k] = 1.

            if max_over_obj:
                kval = MAD.calc_objective(yk, y, G)
            else:
                kval = MAD.calc_surrogate_objective(yk, y, G)
            if kval > argmax_val:
                argmax_val = kval
                argmax_k = k

        ystar = torch.zeros_like(y)
        ystar[argmax_k] = 1.

        return ystar, argmax_val

    @staticmethod
    def oracle_argmax_preserving(G, y, max_over_obj=False):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        if K > 10:
            # return MAD.oracle_argmax_preserving_approx(G, y, max_over_obj)
            return MAD.oracle_argmax_preserving_approx_gpu(G, y, max_over_obj)

        max_k = y.argmax()
        G_np = G.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        ystar = None
        max_val = -1.

        # Generate a set of 1-hot encoded vertices
        # This command produces vertex sets e.g., for K=3: 000, 001, 010, 011, ..., 111
        # Idea is to distribute prob. mass equally over vertices set to 1
        vertices = np.asarray(list(itertools.product([0, 1], repeat=K)), dtype=np.float32)
        # Select vertices where k-th vertex = 1
        vertices = vertices[vertices[:, max_k] > 0]
        # Iterate over these vertices to find argmax k
        for y_extreme in vertices:
            # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
            # Upweigh k-th label by epsilon to maintain argmax label
            y_extreme[max_k] += 1e-5
            # Convert to prob vector
            y_extreme = y_extreme / y_extreme.sum()

            # Doing this on CPU is much faster (I guess this is because we don't need a mem transfer each iteration)
            if max_over_obj:
                kval = MAD.calc_objective_numpy(y_extreme, y_np, G_np)
            else:
                kval = MAD.calc_surrogate_objective_numpy(y_extreme, y_np, G_np)

            if kval > max_val:
                max_val = kval
                ystar = y_extreme

        ystar = torch.tensor(ystar).to(G.device)
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_argmax_preserving_approx_gpu(G, y, max_over_obj=False, max_iters=1024):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        max_k = y.argmax().item()
        G_np = G.detach()
        y_np = y.detach().clone()

        # To prevent underflow
        y_np += 1e-8
        y_np /= y_np.sum()

        ystar = None
        max_val = -1.
        niters = 0.

        # By default, we have (K-1)! vertices -- this does not scale when K is large (e.g., CIFAR100).
        # So, perform the search heuristically.
        # Search strategy used:
        #   a. find k which maximizes objective
        #   b. fix k
        #   c. repeat
        fixed_verts = [max_k, ]  # Grow this set

        while niters < max_iters:
            y_prev_extreme = torch.zeros(K)
            y_prev_extreme[fixed_verts] = 1.

            # Find the next vertex extreme
            k_list = np.array(sorted((set(range(K)) - set(fixed_verts))), dtype=int)
            if len(k_list)==0:
                break
            y_extreme_batch = []
            for i, k in enumerate(k_list):
                y_extreme = y_prev_extreme.clone().detach()
                # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
                y_extreme[k] = 1.

                # Upweigh k-th label by epsilon to maintain argmax label
                y_extreme[max_k] += 1e-5
                # Convert to prob vector
                y_extreme = y_extreme / y_extreme.sum()

                y_extreme_batch.append(y_extreme)

            y_extreme_batch = torch.stack(y_extreme_batch).transpose(0, 1).to(G_np.device)
            assert y_extreme_batch.size() == torch.Size([K, len(k_list)]), '{} != {}'.format(y_extreme_batch.size(),
                                                                                             (K, len(k_list)))
            B = y_extreme_batch.size(1)

            y_np_batch = torch.stack([y_np.clone().detach() for i in range(B)]).transpose(0, 1)

            kvals = MAD.calc_objective_batched(y_extreme_batch, y_np_batch, G_np)

            max_i = kvals.argmax().item()
            max_k_val = kvals.max().item()

            if max_k_val > max_val:
                max_val = max_k_val
                ystar = y_extreme_batch[:, max_i]

            next_k = k_list[max_i]
            fixed_verts.append(next_k)

            niters += B

        try:
            ystar = ystar.clone().detach()
        except AttributeError:
            import ipdb;
            ipdb.set_trace()
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_argmax_preserving_approx(G, y, max_over_obj=False, max_iters=1024):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        max_k = y.argmax()
        G_np = G.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        ystar = None
        max_val = -1.
        niters = 0.

        # By default, we have (K-1)! vertices -- this does not scale when K is large (e.g., CIFAR100).
        # So, perform the search heuristically.
        # Search strategy used:
        #   a. find k which maximizes objective
        #   b. fix k
        #   c. repeat
        fixed_verts = np.array([max_k, ], dtype=int)  # Grow this set
        while niters < max_iters:
            y_prev_extreme = np.zeros((K,), dtype=np.float32)
            y_prev_extreme[fixed_verts] = 1.

            # Find the next vertex extreme
            k_list = np.array(sorted((set(range(K)) - set(fixed_verts))), dtype=int)
            y_extreme_batch = []
            for i, k in enumerate(k_list):
                y_extreme = y_prev_extreme.copy()
                # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
                y_extreme[k] = 1.

                # Upweigh k-th label by epsilon to maintain argmax label
                y_extreme[max_k] += 1e-5
                # Convert to prob vector
                y_extreme = y_extreme / y_extreme.sum()

                y_extreme_batch.append(y_extreme)

            y_extreme_batch = np.array(y_extreme_batch).T
            assert y_extreme_batch.shape == (K, len(k_list)), '{} != {}'.format(y_extreme_batch.shape, (K, len(k_list)))
            B = y_extreme_batch.shape[1]

            y_np_batch = np.stack([y_np.copy() for i in range(B)]).T.astype(np.float32)

            kvals = MAD.calc_objective_numpy_batched(y_extreme_batch, y_np_batch, G_np)

            max_i = np.argmax(kvals)
            max_k_val = np.max(kvals)

            if max_k_val > max_val:
                max_val = max_k_val
                ystar = y_extreme_batch[:, max_i]

            next_k = k_list[max_i]
            fixed_verts = np.concatenate((fixed_verts, [next_k, ]), axis=0)

            niters += B

        ystar = torch.tensor(ystar).to(G.device)
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_batch(G,y,epsilon,oracle_type='extreme',opt_method='interior-point',tolerance=None):
        """
        use linear programing to solve the approximate optimization problem:
        max_y'       y^TGG^Ty'
        subject to   Ay'=[1,1,1,...,1]^T
                     z_i>= y'_i-y_i
                     z_i>= -y'_i+y_i
                     \sum_i z_i<=\epsilon
        (optional)   argmax y_i = argmax y'_i for all i = 1,...,n
        """
        assert oracle_type in ['extreme','argmax'], 'Not a supported oracle type for batch input'
        D = G.shape[1]
        n,K = y.shape
        
        
        with torch.no_grad():
            # c = y^TGG^T
            GG = G.matmul(G.transpose(0,1))# +1e-6*torch.eye(n*K).to(G)
            
            c = (y.view(1,-1)).matmul(GG)
            c = c.cpu().numpy()
            c = np.concatenate([c,np.zeros(n*K)],axis=None) # dimension of c should be 2nk
        
        # Simplex Constraint
        
        A_eq = np.zeros([n,2*n*K]) # A_eq: n x 2nK
        b_eq = np.ones(n)
        for i in range(n):
            A_eq[i,i*K:(i+1)*K]=1.0
        
        if tolerance is not None and tolerance>0:
            # relax equation constraints to unequation constraints
            A_eq = np.concatenate([A_eq,-A_eq],axis = 0)
            b_eq = np.concatenate([b_eq+tolerance,tolerance-b_eq],axis=None)
        
        bounds=[0.0,None]

        # Utility Constraint: L1 bounded perturbation
        A_ub = np.zeros([2*n*K,2*n*K])
        b_ub = np.concatenate([y.detach().cpu().numpy(),-y.detach().cpu().numpy()],axis=None)
        for i in range(n*K):
            # y'_i-z_i<=y_i
            A_ub[i,i]=1.
            A_ub[i,i+n*K]=-1.
            # -y'_i-z_i<=-y_i
            A_ub[i+n*K,i]=-1.
            A_ub[i+n*K,i+n*K]=-1.
        # \sum_i z_i<=epsilon
        Z_ub = np.concatenate([np.zeros([1,n*K]),np.ones([1,n*K])],axis=1)
        A_ub = np.concatenate([A_ub,Z_ub],axis=0) # A_ub: (2nK+1) x 2nK
        b_ub = np.append(b_ub,epsilon)

        if tolerance is not None and tolerance>0:
            # relax equation constraints to unequation constraints
            A_ub = np.concatenate([A_ub,A_eq],axis=0)
            b_ub = np.concatenate([b_ub,b_eq],axis=None)
        
        # MAD-argmax: Preserving top-1 label
        if oracle_type == 'lp_argmax':
            A_ub_argmax = np.zeros([n*(K-1),2*n*K])
            b_ub_argmax = np.zeros(n*(K-1))
            max_idxs = (y.argmax(dim=1)).detach().cpu().numpy()
            for i in range(n):
                A_ub_argmax[i*(K-1):(i+1)*(K-1),i*K+max_idxs[i]]=-1.
                t = 0
                for j in range(K):
                    if j == max_idxs[i]:
                        continue
                    # y'_k^i>=y'_k'^i with k=argmax y^i and k'!=k
                    A_ub_argmax[i*(K-1)+t,i*K+j]=1.
                    t += 1
            A_ub = np.concatenate([A_ub,A_ub_argmax],axis=0) # A_ub: (3nK+1-n) x (2nK)
            b_ub = np.concatenate([b_ub,b_ub_argmax],axis=None)


        # A_lower_bound = -np.eye(n*K,2*n*K)
        # A_upper_bound = np.eye(n*K,2*n*K)
        # b_lower_bound = np.zeros(n*K)
        # b_upper_bound = np.ones(n*K)
        # A_ub = np.concatenate([A_ub,A_lower_bound,A_upper_bound],axis=0)
        # b_ub = np.concatenate([b_ub,b_lower_bound,b_upper_bound],axis=None)
        
        warnings.filterwarnings("ignore")# disable warning output
        # solve the linear programming
        if tolerance is not None and tolerance>0:
            # use interior-point method for faster solving
            # use Simplex method to avoid ill-conditioned matrix while solving 
            res = linprog(c=c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method=opt_method)
        
        else:
            # use interior-point method for faster solving
            # use Simplex method to avoid ill-conditioned matrix or other warning while solving 
            res = linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method=opt_method)
            
        warnings.filterwarnings("default")
        y_star = torch.tensor(res.x[:n*K]).to(y)
        y_star = y_star.view(n,K)
        min_val = torch.tensor(res.fun)

        return y_star,min_val

    @staticmethod
    def oracle_batch_pulp(G,y,epsilon,batch_constraint=False,oracle_type='lp_argmax',tolerance=None):
        """
        use linear programing to solve the approximate optimization problem:
        max_y'       y^TGG^Ty'
        subject to   Ay'=[1,1,1,...,1]^T
                     z_i>= y'_i-y_i
                     z_i>= -y'_i+y_i
                     \sum_i z_i<=\epsilon
        (optional)   argmax y_i = argmax y'_i for all i = 1,...,n
        """
        assert oracle_type in ['lp_extreme','lp_argmax'], 'Not a supported oracle type for batch input'

        n,K = y.shape
        y_ori = y
        # print(G.shape)
        
        with torch.no_grad():
            # c = y^TGG^T
            if G is not None:
                GG = G.matmul(G.transpose(0,1))# +1e-6*torch.eye(n*K).to(G)
                c = (y.view(1,-1)).matmul(GG)
                c = c.flatten().cpu().numpy()
            else:
                c = y.flatten().cpu().numpy()
            
            # c = np.concatenate([c,np.zeros(n*K)],axis=None) # dimension of c should be 2nk
        y = y.detach().cpu().numpy()

        solver = PULP_CBC_CMD(msg=False)
        prob = LpProblem("MAD-Batch",LpMinimize)
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

        # Argmax Preserving
        if oracle_type == 'lp_argmax':
            max_idxs = np.argmax(y,axis=1)
            for i in range(n):
                for j in range(K):
                    if j != max_idxs[i]:
                        prob += ys[i][max_idxs[i]]-ys[i][j]>=1e-4, "y'[{0},{1}]>=y'[{0},{2}]".format(i,max_idxs[i],j)
        
        
        prob.solve(solver=solver)
        y_star = np.zeros([n,K])
        min_val = torch.tensor(value(prob.objective))
        for i in range(n):
            for j in range(K):
                y_star[i,j]=ys[i][j].varValue
        y_star = torch.tensor(y_star).to(y_ori)
        
        return y_star,min_val
        

    @staticmethod
    def oracle_rand(G, y):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        rand_k = np.random.randint(low=0, high=K)

        ystar = torch.zeros_like(y)
        ystar[rand_k] = 1.
        return ystar, torch.tensor(-1.)

    @staticmethod
    def oracle_argmin(G, y):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        argmin_k = y.argmin().item()
        ystar = torch.zeros_like(y)

        ystar[argmin_k] = 1.
        return ystar, torch.tensor(-1.)


    @staticmethod
    def project_ydist_constraint(delta, epsilon, ydist, y=None):
        assert len(delta.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']

        device = delta.device
        K, = delta.shape

        assert delta.shape == torch.Size([K, ])
        if ydist == 'l1':
            delta_numpy = delta.detach().cpu().numpy()
            delta_projected = euclidean_proj_l1ball(delta_numpy, s=epsilon)
            delta_projected = torch.tensor(delta_projected)
        elif ydist == 'l2':
            delta_projected = epsilon * delta / delta.norm(p=2).clamp(min=epsilon)
        elif ydist == 'kl':
            raise NotImplementedError()
        delta_projected = delta_projected.to(device)
        return delta_projected

    @staticmethod
    def project_simplex_constraint(ytilde):
        assert len(ytilde.shape) == 1, 'Does not support batching'
        K, = ytilde.shape
        device = ytilde.device

        ytilde_numpy = ytilde.detach().cpu().numpy()
        ytilde_projected = euclidean_proj_simplex(ytilde_numpy)
        ytilde_projected = torch.tensor(ytilde_projected)
        ytilde_projected = ytilde_projected.to(device)
        return ytilde_projected

    @staticmethod
    def closed_form_alpha_estimate(y, ystar, ydist, epsilon):
        assert y.shape == ystar.shape, 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert len(y.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']
        K, = y.shape

        if ydist == 'l1':
            p = 1.
        elif ydist == 'l2':
            p = 2.
        else:
            raise ValueError('Only supported for l1/l2')
        alpha = epsilon / ((y - ystar).norm(p=p) + 1e-7)
        alpha = alpha.clamp(min=0., max=1.)
        return alpha


    def linesearch(self,G, y, ystar, ydist, epsilon, closed_alpha=True):
        """
        Let h(\alpha) = (1 - \alpha) y + \alpha y*
        Compute \alpha* = argmax_{\alpha} h(\alpha)
        s.t.  dist(y, h(\alpha)) <= \epsilon

        :param G:
        :param y:
        :param ystar:
        :return:
        """
        K, D = G.shape
        assert y.shape == ystar.shape == torch.Size([K, ]), 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert ydist in ['l1', 'l2', 'kl']

        # Definition of h
        h = lambda alpha: (1 - alpha) * y + alpha * ystar

        # Short hand for distance function
        dist_func = lambda y1, y2: self.calc_distance(y1, y2, ydist)

        if ydist in ['l1', 'l2'] and closed_alpha:
            # ---------- Optimally compute alpha
            alpha = MAD.closed_form_alpha_estimate(y, ystar, ydist, epsilon)
            ytilde = h(alpha)
        else:
            # ---------- Bisection method
            alpha_low, alpha_high = 0., 1.
            h_low, h_high = h(alpha_low), h(alpha_high)

            # Sanity check
            feasible_low = dist_func(y, h_low) <= epsilon
            feasible_high = dist_func(y, h_high) <= epsilon
            assert feasible_low or feasible_high

            if feasible_high:
                # Already feasible. Our work here is done.
                ytilde = h_high
                delta = ytilde - y
                return delta
            else:
                ytilde = h_low

            # Binary Search
            for i in range(15):
                alpha_mid = (alpha_low + alpha_high) / 2.
                h_mid = h(alpha_mid)
                feasible_mid = dist_func(y, h_mid) <= epsilon

                if feasible_mid:
                    alpha_low = alpha_mid
                    ytilde = h_mid
                else:
                    alpha_high = alpha_mid

        delta = ytilde - y
        return delta

    @staticmethod
    def greedy(G, y, ystar):
        NotImplementedError()



    def projections(self,G, y, ystar, epsilon, ydist, max_iters=100):
        K, D = G.shape
        assert y.shape == ystar.shape == torch.Size([K, ]), 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert ydist in ['l1', 'l2', 'kl']

        ytilde = ystar
        device = G.device

        for i in range(max_iters):
            # (1) Enforce distance constraint
            delta = ytilde - y
            delta = MAD.project_ydist_constraint(delta, epsilon, ydist).to(device)
            ytilde = y + delta

            # (2) Project back into simplex
            ytilde = MAD.project_simplex_constraint(ytilde).to(device)

            # Break out if constraints are met
            if self.is_in_dist_ball(y, ytilde, ydist, epsilon) and MAD.is_in_simplex(ytilde):
                break

        delta = ytilde - y
        return delta

    def calc_delta(self, x, y):
        # Jacobians G
        if self.disable_jacobian or self.oracle in ['random', 'argmin']:
            G = None
        else:
            # _start = time.time()
            G = MAD.compute_jacobian_nll(x, self.model_adv_proxy, device=self.device,max_grad_layer=self.max_grad_layer)
            # _end = time.time()
            # self.jacobian_times.append(_end - _start)
            # if np.random.random() < 0.05:
            #     print('mean = {:.6f}\tstd = {:.6f}'.format(np.mean(self.jacobian_times), np.std(self.jacobian_times)))
            # # print(_end - _start)
            if self.D is None:
                self.D = G.shape[1]

        if self.oracle not in ['lp_argmax','lp_extreme']:
            # y* via oracle
            assert len(y)==1, "Does not support batching in original MAD!"
            y = y.flatten()
            if self.oracle == 'random':
                ystar, ystar_val = self.oracle_rand(G, y)
            elif self.oracle == 'extreme':
                ystar, ystar_val = self.oracle_extreme(G, y, max_over_obj=self.objmax)
            elif self.oracle == 'argmin':
                ystar, ystar_val = self.oracle_argmin(G, y)
            elif self.oracle == 'argmax':
                ystar, ystar_val = self.oracle_argmax_preserving(G, y, max_over_obj=self.objmax)
            else:
                raise ValueError()
            
            # y* maybe outside the feasible set - project it back
            if self.optim == 'linesearch':
                delta = self.linesearch(G, y, ystar, self.ydist, self.epsilon)
            elif self.optim == 'projections':
                delta = self.projections(G, y, ystar, self.ydist, self.epsilon)
            elif self.optim == 'greedy':
                raise NotImplementedError()
            else:
                raise ValueError()

            # Calc. final objective values
            ytilde = y + delta
            objval = self.calc_objective(ytilde, y, G)
            objval_surrogate = self.calc_surrogate_objective(ytilde, y, G)
        else:# for lp approx
            # ystar,ystar_val = self.oracle_batch(G,y,self.epsilon,self.oracle,opt_method='revised simplex',tolerance=None)
            ystar,ystar_val = self.oracle_batch_pulp(G,y,self.epsilon,self.batch_constraint,self.oracle,tolerance=None)
            delta = ystar-y
            objval = ystar_val
            objval_surrogate = torch.tensor(0.0)

        return delta, objval, objval_surrogate


    def __call__(self, x, stat=True, return_origin=False):
        TypeCheck.multiple_image_blackbox_input_tensor(x)  # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1).detach()
            if stat:
                self.call_count += x.shape[0]


        # with torch.enable_grad():
        if self.epsilon > 0.:
            delta, objval, sobjval = self.calc_delta(x, y_v)
        else:
            delta = torch.zeros_like(y_v)
            objval, sobjval = torch.tensor(0.), torch.tensor(0.)

        y_prime = y_v + delta

        # ---------------------- Sanity checks
        # ---------- 1. No NaNs
        assert torch.isnan(delta).sum().item() == 0., ' y = {}\n delta = {}'.format(y_v, delta)
        # ---------- 2. Constraints are met
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
                                    objval.cpu().detach().numpy(), sobjval.cpu().detach().numpy()))

            # y_prime.append(y_prime_i)

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

        # y_prime = torch.stack(y_prime)

        if return_origin:
            return y_prime,y_v
        else:
            return y_prime

    def get_yprime(self,y,x_info=None):
        if self.oracle not in ['lp_argmax','lp_extreme']:
            n,K = y.shape
            yprime = torch.zeros_like(y)
            if x_info is None:
                G = torch.eye(K).to(y)
            else:
                G = x_info
            if self.D is None:
                self.D = G.shape[1]
            # y* via oracle
            for i in range(n):
                if self.oracle == 'random':
                    ystar, ystar_val = self.oracle_rand(G, y[i])
                elif self.oracle == 'extreme':
                    ystar, ystar_val = self.oracle_extreme(G, y[i], max_over_obj=self.objmax)
                elif self.oracle == 'argmin':
                    ystar, ystar_val = self.oracle_argmin(G, y[i])
                elif self.oracle == 'argmax':
                    ystar, ystar_val = self.oracle_argmax_preserving(G, y[i], max_over_obj=self.objmax)
                else:
                    raise ValueError()
                
                # y* maybe outside the feasible set - project it back
                if self.optim == 'linesearch':
                    delta = self.linesearch(G, y[i], ystar, self.ydist, self.epsilon)
                elif self.optim == 'projections':
                    delta = self.projections(G, y[i], ystar, self.ydist, self.epsilon)
                elif self.optim == 'greedy':
                    raise NotImplementedError()
                else:
                    raise ValueError()

                # Calc. final objective values
                yprime[i] = y[i] + delta
            
        else:# for lp approx
            # ystar,ystar_val = self.oracle_batch(G,y,self.epsilon,self.oracle,opt_method='revised simplex',tolerance=None)
            yprime,_ = self.oracle_batch_pulp(None,y,self.epsilon,self.batch_constraint,self.oracle,tolerance=None)

        return yprime

    def get_xinfo(self,x):
        if self.disable_jacobian or self.oracle in ['random', 'argmin']:
            G = None
        else:
            G = MAD.compute_jacobian_nll(x, self.model_adv_proxy, device=self.device,max_grad_layer=self.max_grad_layer)
                
            # n = len(x)
            # G = G.reshape([n,-1,G.size(1)])
        return G
