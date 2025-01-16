from turtle import distance
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import TensorDataset,DataLoader
from tqdm import tqdm
import csv
import os.path as osp
import os
import json
import pickle
import numpy as np

import torch.multiprocessing
from torch.multiprocessing import Process,Manager

from defenses.utils.model import soft_cross_entropy
from defenses.victim import AM
import defenses.models.zoo as zoo
from defenses import datasets

import pdb

numclasses_to_nn = {
    10:[128,64],
    43:[512,256],
    100:[1024,512],
    200:[2048,1024],
    256:[2048,1024]
}


class Recover_NN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        hidden_layer=numclasses_to_nn[num_classes]
        self.fc1 = nn.Linear(num_classes,hidden_layer[0])
        self.fc2 = nn.Linear(hidden_layer[0],hidden_layer[1])
        self.fc3 = nn.Linear(hidden_layer[1],num_classes)

        self.bn1 = nn.BatchNorm1d(hidden_layer[0])
        self.bn2 = nn.BatchNorm1d(hidden_layer[1])
        self.bn3 = nn.BatchNorm1d(num_classes)

    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.fc1(x)),0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)),0.2)
        return F.softmax(x,dim=1)
    



class Table_Recover():
    max_sample_size=5000000
    def __init__(self,blackbox,table_size=1000000,batch_size=1,epsilon=None,perturb_norm=1,recover_mean=True,recover_norm=2,tolerance=1e-4,concentration_factor=4.0,shadow_path=None,recover_nn=False,recover_proc=1):
        self.table_size = table_size
        self.blackbox = blackbox
        self.num_classes = self.blackbox.num_classes
        self.device = self.blackbox.device
        self.batch_size = batch_size
        self.epsilon=epsilon
        self.perturb_norm = perturb_norm
        self.top1_recover = self.blackbox.top1_preserve
        self.recover_mean=recover_mean
        self.recover_norm = recover_norm
        self.tolerance = tolerance
        self.concentration_factor=concentration_factor
        self.load_nn = False
        if shadow_path is not None and osp.exists(shadow_path):
            self.shadow_generate=True
            self.shadow_path = shadow_path
        else:
            self.shadow_generate=False
        self.recover_nn = bool(recover_nn)

        self.num_proc = recover_proc
        self.true_label_sample,self.perturbed_label_sample = None, None
        if not self.recover_nn:
            self.log_path = osp.join(self.blackbox.out_path, 'recover_distance{}.log.tsv'.format(self.blackbox.log_prefix))
            self.logger = csv.writer(open(self.log_path,'a'),delimiter='\t')
            self.logger.writerow(['call count','recover distance mean', 'recover distance std'])
            self.call_count = 0
        else:
            self.log_path = osp.join(self.blackbox.out_path, 'recover_nn_training.log.tsv')
            self.logger = csv.writer(open(self.log_path,'a'),delimiter='\t')
            self.logger.writerow(['Epoch','Loss', 'L2 Distance'])
    
        

    def generate_lookup_table(self,load_path=None,estimation_set=None,table_size=None,load_nn=False):
        if table_size is None:
            table_size = self.table_size
        if load_path is not None:
            if osp.exists(load_path):
                with open(load_path,'rb') as wf:
                    self.true_label_sample,self.perturbed_label_sample = pickle.load(wf)
                    print("Loaded Existing Table with table length {}!".format(len(self.true_label_sample)))
                    if len(self.true_label_sample)>=table_size:
                        self.true_label_sample,self.perturbed_label_sample = self.true_label_sample[:table_size,:],self.perturbed_label_sample[:table_size,:]
                        table_size = 0
                    else:
                        print("Supplementing existing table with {} samples...".format(table_size-len(self.true_label_sample)))
        if self.true_label_sample is not None:
            table_size -= len(self.true_label_sample)
        if table_size>0:
            print("Building Recover Table! Total Samples Number={}!".format(table_size))
            true_label_sample = []
            x_info_idxs = []
            if self.shadow_generate: # generate true labels with shadow models
                print("Use shadow models for generation!")
                assert estimation_set is not None, "The esitimation set cannot by None when using shadow models for true prediction generation!"
                estimation_input,_ = self.estimate_dir(estimation_set)
                for d in os.listdir(self.shadow_path):
                    if "shadow" in d and osp.exists(osp.join(self.shadow_path,d,'checkpoint.pth.tar')):
                        params_dir = osp.join(self.shadow_path,d,'params.json')
                        with open(params_dir) as f:
                            params = json.load(f)
                        shadow_dataset = params['dataset']
                        shadow_arch = params['model_arch']
                        num_classes = params['num_classes']
                        modelfamily = datasets.dataset_to_modelfamily[shadow_dataset]
                        shadow_model = zoo.get_net(shadow_arch, modelfamily, osp.join(self.shadow_path,d), num_classes=num_classes)
                        shadow_model.to(self.device)
                        shadow_model.eval()
                        for i in range(0,len(estimation_input),self.batch_size):
                            x = estimation_input[i:min(i+self.batch_size,len(estimation_input))].to(self.device)
                            y = F.softmax(shadow_model(x),dim=1).detach().cpu()
                            true_label_sample.append(y)
                        x_info_idxs += list(range(len(estimation_input)))
                true_label_sample = torch.cat(true_label_sample,dim=0)
                table_size = table_size-len(true_label_sample)
                if table_size < 0:# shrink the size of the table with random choice
                    subset = np.random.choice(list(range(len(true_label_sample))),len(true_label_sample)+table_size,replace=False)
                    true_label_sample = true_label_sample[subset]
                    x_info_idxs = [x_info_idxs[i] for i in subset]

            if table_size>0: # supplement the table with true labels generated from dirichlet distribution
                if estimation_set is not None:
                    estimation_input,estimation_label = self.estimate_dir(estimation_set)
                    concentration = self.num_classes*self.concentration_factor
                    alpha = estimation_label*concentration# The std of dirichlet distribution is prop to 1/sqrt(concentration)
                    # x_infos = self.blackbox.get_xinfo(estimation_input)
                else: 
                    estimation_input = None
                    alpha = None
                
                true_label_sample_dir,x_info_idxs_dir = self.get_dirichlet_samples(alpha,table_size)
                if len(true_label_sample)==0:
                    true_label_sample = true_label_sample_dir
                else:
                    true_label_sample = torch.cat([true_label_sample,true_label_sample_dir],dim=0)
                x_info_idxs += x_info_idxs_dir

            if not self.blackbox.require_xinfo: # if the blackbox does not require xinfo for yprime, we disable it in the following procedure.
                estimation_input = None
            if self.num_proc == 1:
                perturbed_label_sample = self.get_perturbed_label_sample(self.blackbox,true_label_sample,estimation_input,x_info_idxs,self.batch_size)
            else:
                perturbed_label_sample = self.get_perturbed_label_sample_parallel(self.blackbox,true_label_sample,estimation_input,x_info_idxs,self.num_proc)
            
            if self.epsilon is not None:
                pert_norm = torch.norm(true_label_sample-perturbed_label_sample,p=self.perturb_norm,dim=1)
                true_label_sample = true_label_sample[pert_norm<=self.epsilon]
                perturbed_label_sample = perturbed_label_sample[pert_norm<=self.epsilon]
            if self.true_label_sample is None or self.perturbed_label_sample is None:
                self.true_label_sample,self.perturbed_label_sample = true_label_sample,perturbed_label_sample
            else:
                self.true_label_sample = torch.cat([self.true_label_sample,true_label_sample.to(self.true_label_sample)],dim=0)
                self.perturbed_label_sample = torch.cat([self.perturbed_label_sample,perturbed_label_sample.to(self.perturbed_label_sample)],dim=0)
 
            print("Recover Table Completed!")
        
            with open(osp.join(self.blackbox.out_path, 'recover_table.pickle'), 'wb') as wf:
                pickle.dump([self.true_label_sample,self.perturbed_label_sample], wf)
        try:
            self.true_label_sample,self.perturbed_label_sample = self.true_label_sample.to(self.device),self.perturbed_label_sample.to(self.device)
        except:
            print("[Warning]: Not enough GPU memory for storing the lookup table, will use cpu instead!")
            # self.device = torch.device('cpu')
            self.true_label_sample,self.perturbed_label_sample = self.true_label_sample.cpu(),self.perturbed_label_sample.cpu()

        if not self.recover_nn:
            if self.top1_recover:
                self.true_top1 = torch.argmax(self.true_label_sample,dim=1).to(self.device)
            
        else:
            self.nn = Recover_NN(self.num_classes)
            print("Generative Model:")
            print(self.nn)

            self.nn.to(self.device)
            model_out_path = self.log_path = osp.join(self.blackbox.out_path, 'recover_nn.pt')
            if osp.exists(model_out_path) and load_nn:
                print("Load existing generative model at "+model_out_path)
                self.nn.load_state_dict(torch.load(model_out_path))
            else:
                print("Training NN for Recovering!")  

                # if recover_nn_cub200.pt not exists, train the model
                if not osp.exists('recover_nn.pt'):
                    self.nn = self.train_recover_nn(self.nn,self.perturbed_label_sample,self.true_label_sample,epoch=20,batch_size=1024,lr=1e-2) 
                    torch.save(self.nn.state_dict(), 'recover_nn.pt')
                else:
                    self.nn.load_state_dict(torch.load('recover_nn.pt'))
                
                torch.save(self.nn.state_dict(), model_out_path)
                #self.nn.load_state_dict(torch.load('recover_nn_caltech256.pt'))

                #pdb.set_trace()
                
            

    def estimate_dir(self,estimation_set):
        if isinstance(estimation_set,str) and osp.exists(estimation_set): # use labels in a transfer set
            print("Estimating Dirichlet Distribution via Lables in '{}'".format(estimation_set))
            with open(estimation_set,'rb') as wf:
                estimation_data = pickle.load(wf)
                estimation_input = torch.cat([torch.tensor(estimation_data[i][0]).reshape([1,-1]) for i in range(len(estimation_data))],dim=0)
                estimation_label = torch.cat([torch.tensor(estimation_data[i][1]).reshape([1,-1]) for i in range(len(estimation_data))],dim=0)
        else: # use labels in a tensor
            try:
                estimation_input = estimation_set[0]
                estimation_label = estimation_set[1].clone().detach()
            except:
                raise RuntimeError("Not a valid estimation set form (must be a path or a list of tensors)") 
        return estimation_input,estimation_label.to(self.device)

    def get_dirichlet_samples(self,alpha=None,table_size=1000000):
        """
        Generate samples from dirichlet distribution
        """
        sample_list = []
        alpha_idxs = []
        # if estimation_set is not None:
        #     concentration = self.num_classes*4
        #     alpha = self.estimate_dir(estimation_set)*concentration
        if alpha is None:# preset alphas
            alpha = [k*torch.ones(self.num_classes).to(self.device)/self.num_classes for k in [1.0,]]
        s = table_size//len(alpha)

        for n,a in enumerate(alpha):
            alpha_idxs += [n]*s
            distribution = Dirichlet(a)
            group_num = s//self.max_sample_size # the maximum size of samples is 5000000 in one generation
            final_group = s%self.max_sample_size
            for _ in range(group_num):
                samples = distribution.sample((self.max_sample_size,)).cpu()
                sample_list.append(samples)
            sample_list.append(distribution.sample((final_group,)).cpu())
            
            # samples = distribution.sample((s,)).cpu()
            # sample_list.append(samples)
        return torch.cat(sample_list,dim=0),alpha_idxs

    def get_uniform_samples(self,table_size=1000000):
        """
        Generate samples uniformly
        """
        raise NotImplementedError("Not implemented uniform sampling")


    @staticmethod
    def get_perturbed_label_sample(blackbox,true_label_sample,xs=None,x_info_idxs=None,batch_size=32,output=None,count=None,proc_idx=None):    
        if count is None:
            pbar = tqdm(total=len(true_label_sample))
        #with tqdm(total=len(true_label_sample)) as pbar:
        if xs is None or x_info_idxs is None:
            perturbed_label_sample = []
            for start_idx in range(0,len(true_label_sample),batch_size):
                end_idx = min([start_idx+batch_size,len(true_label_sample)])
                perturbed_label = blackbox.get_yprime(true_label_sample[start_idx:end_idx,:].to(blackbox.device))
                perturbed_label_sample.append(perturbed_label.detach().to(true_label_sample))
                if count is not None:
                    count.value += len(perturbed_label)
                else:
                    pbar.update(len(perturbed_label))
            if output is not None and proc_idx is not None:
                output[proc_idx] = torch.cat(perturbed_label_sample,dim=0)
                return
            else:
                return torch.cat(perturbed_label_sample,dim=0)
        else:
            assert len(x_info_idxs) == len(true_label_sample), "The length of x_info_idxs must be equal to the length of true_label_sample!" 
            x_info_idxs_set = set(x_info_idxs)
            x_info_idxs = np.array(x_info_idxs)
            perturbed_label_sample = torch.zeros_like(true_label_sample)
            for i in x_info_idxs_set:
                x_i = xs[i].unsqueeze(0)
                index_i = np.arange(len(true_label_sample))[x_info_idxs==i]
                true_label_i = true_label_sample[index_i].to(blackbox.device)
                x_info = blackbox.get_xinfo(x_i) # the x_info could be too large to store, so we consider it one by one
                for start_idx in range(0,len(true_label_i),batch_size):
                    end_idx = min([start_idx+batch_size,len(true_label_i)])
                    perturbed_label = blackbox.get_yprime(true_label_i[start_idx:end_idx,:],x_info = x_info)
                    perturbed_label_sample[index_i[start_idx:end_idx],:] = perturbed_label.detach().to(true_label_sample)
                    if count is not None:
                        count.value += len(perturbed_label)
                    else:
                        pbar.update(len(perturbed_label))
            if output is not None and proc_idx is not None:
                output[proc_idx] = perturbed_label_sample
                return
            else:
                return perturbed_label_sample

                

        


    def get_perturbed_label_sample_parallel(self,blackbox,true_label_sample,xs=None,x_info_idxs=None,num_proc=10):
        print("Generating recover table with %d processes..."%num_proc)
        torch.multiprocessing.set_start_method('spawn',force=True)
        # if hasattr(blackbox,'cpu'):
        #     blackbox.cpu()
        with Manager() as manager:
            proc_data = np.array_split(true_label_sample,num_proc)
            if xs is not None and x_info_idxs is not None:
                assert len(x_info_idxs) == len(true_label_sample), "x_info_idxs must have the same length as true_lable_sample!"
                x_info_idxs_proc = np.array_split(x_info_idxs,num_proc)
                shared_xs = manager.list([x for x in xs]) # use shared memory to reduce memory consumption
            else:
                x_info_idxs_proc = [None]*num_proc
                shared_xs = None
            count = manager.Value('i',0)
            perturbed_label_output = manager.list([None,]*num_proc)
            # for i in range(num_proc):
            #     perturbed_label_output.append(None)
            proc = []
            for i in range(num_proc):
                p = Process(target=Table_Recover.get_perturbed_label_sample,args=(blackbox,proc_data[i],shared_xs,x_info_idxs_proc[i],self.batch_size,perturbed_label_output,count,i))
                proc.append(p)
            for p in proc:
                p.start()
            with tqdm(total=len(true_label_sample)) as pbar:
                prev_count = 0
                while None in perturbed_label_output:
                    current_count = count.value
                    if current_count>prev_count:
                        pbar.update(current_count-prev_count)
                        prev_count=current_count
                
                current_count = sum([len(perturbed_label_output[n]) for n in range(num_proc)])
                if current_count>prev_count:
                    pbar.update(current_count-prev_count)
                    

            
            for p in proc:
                p.join()
                # p.terminate()
            
            perturbed_label_sample = list(perturbed_label_output)
        res = torch.cat(perturbed_label_sample,dim=0)
        print("Ended multiprocessing with total number of samples = %d"%len(res))
        # if hasattr(blackbox,'to_blackbox_device'):
        #     blackbox.to_blackbox_device()
        return res

    def train_recover_nn(self,model,pert_label,true_label,epoch=20,batch_size=128,lr=1e-3):
        dataset = TensorDataset(pert_label,true_label)
        trainloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
        model.train()

        for e in tqdm(range(epoch)):
            total_loss = 0.0
            total_dist = 0.0
            total_iter = 0
            for n,(pl,tl) in enumerate(trainloader):
                total_iter += 1
                pl,tl = pl.to(self.device),tl.to(self.device)
                output = model(pl)
                loss = F.l1_loss(output,tl)*self.num_classes# the l1_loss will reduce all dimension, we recover it to the l1 norm
                total_loss += loss.item()
                total_dist += torch.mean(torch.norm(output.detach()-tl,p=2,dim=1)).item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (n+1)%10 == 0:
                #     print("Iteration: %d\tloss: %f"%(n+1,loss.item()))
            scheduler.step()
            #l2_dist = torch.mean(torch.norm(model(pl).detach()-tl,p=2,dim=1)).cpu().item()
            print("Epoch: {}\tLoss: {:.4f}\tL2 Distance: {:.4f}".format(e+1,total_loss/total_iter,total_dist/total_iter))
            self.logger.writerow([e+1,total_loss/total_iter,total_dist/total_iter])

        
        return model

    def __call__(self, yprime,pbar=None):
        assert yprime.dim()==2, "yprime must be a batch with dim=2"
        yprime = yprime.to(self.device) # all operations in this model is performed on CPU to avoid memory running out
        # if self.load_nn == False:
        #     self.nn.load_state_dict(torch.load('recover_nn.pt'))
        #     self.load_nn = True
        #     print("Load existing generative model at recover_nn.pt")
        if self.recover_nn:
            res = self.nn(yprime).detach()
            #pdb.set_trace()
            # mask=(yprime.argmax(1)!=res.argmax(1))
            # print(yprime.argmax(1)[mask])
            # print(res.argmax(1)[mask])
            # print((yprime.argmax(1)[mask]==9).sum())
            # print(res)
            # max_indices = yprime.argmax(dim=1)
            # output = torch.zeros_like(yprime).to(self.device)
            # output[torch.arange(yprime.size(0)), max_indices] = yprime[torch.arange(yprime.size(0)), max_indices]
            # res = self.nn(output).detach()

        else:
            res = torch.zeros_like(yprime).to(self.device)
            rec_dis = []
            top1_label = torch.argmax(yprime,dim=1)
            for c in range(self.num_classes):
                y = []
                yprime_c = yprime[top1_label==c]
                if len(yprime_c)==0:
                    continue
                if self.top1_recover: # search for the optimal recovery in the same top-1 labels only for efficiency
                    perturbed_label_filtered = self.perturbed_label_sample[self.true_top1==c,:]
                    true_label_filtered = self.true_label_sample[self.true_top1==c,:]
                else:
                    perturbed_label_filtered = self.perturbed_label_sample
                    true_label_filtered = self.true_label_sample
                perturbed_label_filtered,true_label_filtered = perturbed_label_filtered.to(self.device),true_label_filtered.to(self.device)
                for i in range(len(yprime_c)):
                    # return the true lable with the same top-1 label and minimal perturbed distance
                    distances = torch.norm(yprime_c[i]-perturbed_label_filtered,p=self.recover_norm,dim=1)
                    if not self.recover_mean:
                        min_idx = torch.argmin(distances)
                        y.append(true_label_filtered[min_idx,:].unsqueeze(0))
                        rec_dis.append(distances[min_idx])
                    else:
                        tolerance = max([self.tolerance,torch.min(distances).cpu().item()])
                        # if isinstance(self.blackbox,AM) and torch.max(yprime_c[i]).cpu().item()>self.blackbox.defense_fn.delta_list and tolerance>self.tolerance:
                        #     y.append(yprime_c[i].unsqueeze(0))
                        #     rec_dis.append(torch.tensor(0.0).to(yprime))
                        # else:
                        y.append(torch.mean(true_label_filtered[distances<=tolerance,:],dim=0,keepdim=True))
                        rec_dis.append(torch.mean(distances[distances<=tolerance]))
                    if pbar is not None:
                        pbar.update(1)
                
                res[top1_label==c]=torch.cat(y,dim=0)
            # if isinstance(self.blackbox,AM):
            #     max_yprime,_ = torch.max(yprime,dim=1)
            #     id_idx = max_yprime>self.blackbox.defense_fn.delta_list
            #     od_idx = max_yprime<=self.blackbox.defense_fn.delta_list
            #     mean_mis_info = torch.mean(yprime[od_idx],dim=0)
            #     std_dis = torch.mean(torch.norm(yprime[od_idx]-mean_mis_info,p=2,dim=1))
            #     print("Std of misinformation:",std_dis)
            #     print("id_idx length:",torch.sum(id_idx))
            #     print("Minimum maximum value:",torch.min(max_yprime))
            self.call_count += len(yprime)
            mean_rec_dis = torch.mean(torch.tensor(rec_dis)).cpu().item()
            std_rec_dis = torch.std(torch.tensor(rec_dis)).cpu().item()
            self.logger.writerow([self.call_count,mean_rec_dis,std_rec_dis])
            
        
        return res
            

