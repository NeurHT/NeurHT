import numpy as np
import pickle
import os.path as osp
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Dirichlet
from copy import deepcopy
# import sys
# import os
# sys.path.append(os.getcwd())
from defenses import datasets
from tqdm import tqdm

from pulp import *


# from defenses.victim import *

class incremental_kmeans():
    def __init__(self,blackbox,epsilon,ydist='l1',optim=False,trainingset_name=None,frozen = False,ordered_quantization=True,kmean=False,buffer_size=None):
        self.blackbox = blackbox
        self.model = self.blackbox.model
        self.device = self.blackbox.device
        self.out_path = self.blackbox.out_path
        self.num_classes = self.blackbox.num_classes
        self.log_path = self.blackbox.log_path
        self.log_prefix = self.blackbox.log_prefix
        self.require_xinfo = self.blackbox.require_xinfo
        self.top1_preserve = self.blackbox.top1_preserve
        
        self.epsilon = epsilon
        if ydist == 'l1':
            self.norm = 1
        elif ydist == 'l2':
            self.norm = 2
        else:
            raise RuntimeError("Not supported distance metrics for y distance!")
        self.optim=bool(optim)
        self.frozen = bool(frozen)
        self.ordered = bool(ordered_quantization)
        self.kmean = bool(kmean)
        self.label_buffer = buffer_size//self.num_classes if buffer_size is not None else np.inf
        self.labels = []
        self.centroids = []

        if self.out_path is not None:
            centroids_path = osp.join(self.out_path,'quantization_centroids.pickle')
            if osp.exists(centroids_path): # load labels directly
                print("Loading centroids from "+centroids_path) 
                with open(centroids_path,'rb') as f:
                    self.centroids = pickle.load(f)
            
            label_path = osp.join(self.out_path,'quantization_label.pickle')
            if osp.exists(label_path):
                print("Loading labels from "+label_path)
                with open(label_path,'rb') as f:
                    self.labels = pickle.load(f)
        
        if len(self.labels) == 0 and trainingset_name is not None: # generate labels from blackbox
            modelfamily = datasets.dataset_to_modelfamily[trainingset_name]
            transform_type = 'test'
            transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
            trainingset = datasets.__dict__[trainingset_name](train=False, transform=transform)
            dataloader = DataLoader(trainingset,batch_size=32,num_workers=4,shuffle=False)
            print("training quantizer with training set "+trainingset_name)
            self.labels = [[] for _ in range(self.num_classes)]
            for data,label in tqdm(dataloader):
                y_prime = self.blackbox(data,stat=False)
                for n,l in enumerate(label.numpy()):
                    self.labels[l].append(y_prime[n,:].reshape([1,-1]))
            for l in range(self.num_classes):
                self.labels[l] = torch.cat(self.labels[l],dim=0)
                self.labels[l] = self.labels[l][max(0,len(self.labels[l])-self.label_buffer):]
                
            with open(label_path,'wb') as f:
                pickle.dump([label.cpu() for label in self.labels],f)
        
        if len(self.centroids) == 0 and len(self.labels) > 0:
            for l in range(len(self.labels)):
                self.centroids.append(self.cluster(self.labels[l]))
                

            with open(centroids_path,'wb') as f:
                pickle.dump([centroid.cpu() for centroid in self.centroids],f)
            
        for l in range(len(self.centroids)):
            print("Initialized quantizer: label %d has %d centroids!"%(l,len(self.centroids[l])))

        
        for n in range(len(self.labels)):
            self.labels[n] = self.labels[n].to(self.blackbox.device)
            self.centroids[n] = self.centroids[n].to(self.blackbox.device)

            
        
        if len(self.centroids)==0:    
            print("[Warning]: Using empty centroids as initialization!")

        self.queries = []
        self.quantize_queries = []
        self.call_count = 0
        if self.log_path is not None:
            self.quantize_log_path = self.log_path.replace('distance','quantize_distance')
            self.centroids_log_path = self.log_path.replace('distance','centroid')
            
            if not osp.exists(self.quantize_log_path):
                with open(self.quantize_log_path, 'w') as wf:
                    columns = ['call_count', 'l1_max', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                    wf.write('\t'.join(columns) + '\n')
            if not osp.exists(self.centroids_log_path):
                with open(self.centroids_log_path,'w') as wf:
                    columns = ["call_count",]+ ["label %d"%i for i in range(self.num_classes)]
                    wf.write('\t'.join(columns) + '\n')
        else:
            self.quantize_log_path = None
    
    def set_centroids(self,centroids):
        self.centroids = deepcopy(centroids)

    def get_cluster_centroid(self,cluster_data):
        centroid = torch.mean(cluster_data,dim=0,keepdim=True)
        # if self.norm == 2:
        #     # For l2 norm, we can get the centroid with the mean directly
        #     centroid = torch.mean(cluster_data,dim=0,keepdim=True)
        
        # elif self.norm == 1: 
        #     # For l1 norm, we have two methods: approx and exact
        #     data = cluster_data.detach().cpu().numpy()
        #     max_idx = np.argmax(data[0,:])
        #     n,K = data.shape
        #     solver = PULP_CBC_CMD(msg=False)
        #     if self.optim == 'approx':
        #         # solve the centroids with least l1 norm sum approximately
        #         # if no simplex constraints, the median will be the optimal solution
        #         median = np.median(data,axis=0) 
                
        #         # project the median back to the simplex constraints with minimal l1 distance
        #         prob = LpProblem("simplex projection",LpMinimize)

        #         # build variable
        #         c = LpVariable.dicts("centroid",list(range(K)),lowBound=0.0,upBound=1.0)
        #         z = LpVariable.dicts("absolutevalue",list(range(K)),lowBound=0.0)
                
        #         # add objective function
        #         # minimize l1 perturbation
        #         prob += lpSum([z[i] for i in range(K)])

        #         # add constraints
        #         for i in range(K):
        #             # slack variable constraint: z >= |c-median|
        #             prob += z[i]-c[i]>=-median[i], "z[{}]>=c[{}]-median[{}]".format(i,i,i)
        #             prob += z[i]+c[i]>=median[i], "z[{}]>=median[{}]-c[{}]".format(i,i,i)
                    
        #             # argmax constraint
        #             if i != max_idx:
        #                 prob += c[max_idx]-c[i]>=1e-4, "c[{}]>=c[{}]".format(max_idx,i)
                
        #         # simplex constraint
        #         prob += lpSum([c[i] for i in range(K)])==1.0, "Simplex Constraint"

        #         prob.solve(solver)
        #         centroid = np.zeros([1,K])
        #         for i in range(K):
        #             centroid[0,i]=c[i].varValue
                
        #         centroid = torch.tensor(centroid).to(cluster_data)
                
        #     elif self.optim == 'exact':
        #         # solve the controid with exactly the lowest l1 distance sum
        #         print("[Warning]: Solving for exact l1 centroid, which may take a long time!")

        #         prob = LpProblem("L1 Centroid",LpMinimize)

        #         # build variable
        #         c = LpVariable.dicts("centroid",list(range(K)),lowBound=0.0,upBound=1.0)
        #         z = LpVariable.dicts("absolutevalue",list(range(n*K)),lowBound=0.0)

        #         # add objective function
        #         # minimize total l1 distance
        #         prob += lpSum([z[i] for i in range(n*K)])

        #         # add constraints
        #         for i in range(n):
        #             for j in range(K):
        #                 # slack variable constraints: z[i] >= |data[i]-c|
        #                 prob += z[i*K+j]+c[j]>=data[i,j], "z[{0},{1}]>=data[{0},{1}]-c[{1}]".format(i,j)
        #                 prob += z[i*K+j]-c[j]>=-data[i,j], "z[{0},{1}]>=c[{1}]-data[{0},{1}]".format(i,j)

        #         # argmax constraint
        #         for j in range(K):
        #             if j != max_idx:
        #                 prob += c[max_idx]-c[j]>=1e-4, "c[{}]>=c[{}]".format(max_idx,j)

        #         # simplex constraint
        #         prob += lpSum([c[j] for j in range(K)])==1.0, "Simplex Constraint"

        #         prob.solve(solver)
        #         centroid = np.zeros([1,K])
        #         for j in range(K):
        #             centroid[0,j]=c[j].varValue
                
        #         centroid = torch.tensor(centroid).to(data)
            
        #     else:
        #         raise RuntimeError("Not recognized optimization method: "+self.optim)

        # else:
        #     raise RuntimeError("Not supported ydist")
        return centroid

    def quantize(self,input,centroids):
        # print(input.shape)
        if len(input.shape)==1:
            input = input.reshape([1,-1]) # make it as N x D
        
        
        distance_matrix = []
        for i in range(len(centroids)):
            distance_matrix.append(torch.norm(input-centroids[i,:].reshape([1,-1]),p=self.norm,dim=1,keepdim=True))
        distance_matrix = torch.cat(distance_matrix,dim=1)
        distance,cent_idxs = torch.min(distance_matrix,dim=1)
        return distance,cent_idxs

    def ordered_quantize(self,prediction,centroids,incremental=True):
        if prediction.dim()>1:
            prediction = prediction.reshape([-1,])
        assert len(prediction) == self.num_classes, "only support single input"
        if len(centroids)==0 and not incremental:
            return None
        distances = torch.norm(centroids-prediction,p=self.norm,dim=1)
        valid_cent = torch.arange(len(centroids),device=self.device)[distances<=self.epsilon]
        if len(valid_cent)==0:
            # all centroids are far away from the new input, add new centroid
            if incremental:
                new_centroid = self.get_new_centroid(prediction,centroids,opt=self.optim)
                centroids = torch.cat([centroids,new_centroid],dim=0)
                distance = torch.norm(new_centroid-prediction,p=self.norm,dim=1)[0]
                #print(distances,distance)
                cent_idxs = len(centroids)-1
            else:# return the closest centroid
                distance,cent_idxs = torch.min(distances,dim=0)
        else:
            # use the first centroid that satisfy the utility constraint to avoid returned result change for the same prediction
            cent_idxs = valid_cent[0]
            distance = distances[cent_idxs]
        return distance,cent_idxs,centroids

    def get_new_centroid(self,outlier,centroids,opt=False):
        """
        given existing centroids and an outlier point, generate a new centroid far away from exiting centroids but contain the outlier
        """
        if opt:
            # randomly sample some points near the outlier and select the one with the largest total distance from existing centroids
            s = self.num_classes*10
            Dist = Dirichlet(outlier.reshape([-1,])*s) # use concentration s to make the std of deviation approximately epsilon/sqrt(n)
            samples = Dist.sample([1000,])
            samples = samples[torch.norm(samples-outlier,p=self.norm,dim=1)<=self.epsilon] # discard all samples outside the constraint
            max_class = torch.argmax(samples,dim=1)
            samples = samples[max_class==torch.argmax(outlier)]# preserve the top-1 labels
            if len(samples)>0:
                max_total_dist = 0.0
                max_idxs = -1
                for i in range(len(samples)):
                    total_dist = torch.sum(torch.norm(centroids-samples[i],p=self.norm,dim=1))
                    if total_dist>max_total_dist:
                        max_total_dist = total_dist
                        max_idxs = i
                new_centroid = samples[max_idxs]
            else:# no close centroids sampled
                new_centroid = outlier
        else:
            new_centroid = outlier
        
        return new_centroid.reshape([1,-1])

    def k_means(self,data,inital_centroids,tolerance=1e-3,max_iter=100):
        centroids = inital_centroids
        for _ in range(max_iter):
            new_centroids = deepcopy(centroids)
            quantize_distances,cent_idxs = self.quantize(data,centroids) # cluster
            for c in range(len(centroids)):
                if torch.sum(cent_idxs==c)>0: # skip empty clusters and do not update them
                    new_centroids[c] = self.get_cluster_centroid(data[cent_idxs==c]) # mean/median
            max_move = torch.max(torch.norm(new_centroids-centroids,p=2,dim=1))
            if max_move<tolerance:
                break
            else:
                centroids = new_centroids
        
        # delete empty clusters
        nonempty_cluster = list(set(cent_idxs.cpu().numpy().tolist()))
        centroids = centroids[nonempty_cluster,:]
        quantize_distances,cent_idxs = self.quantize(data,centroids)
            
        
        return quantize_distances,centroids

    def cluster(self,data,init_cluster=None):
        """
        perform cluster with data and return centroids
        """
        if len(data)==0:
            return []
        
        #print("Reclustering!")
        if init_cluster is None:
            # use mean of all data as initialization
            centroids = self.get_cluster_centroid(data)
        else:
            # use the old centroids as the initialization to accelerate the clustering
            centroids = init_cluster
        
        while True:
            val_distance,centroids = self.k_means(data,centroids)
            #val_distance, _ = self.quantize(data,centroids)
            #mean_dis = torch.mean(val_distance,dim=0)
            max_dis,max_idx = torch.max(val_distance,dim=0)

            if max_dis>self.epsilon: # keep adding new centroids until the constraint is satisfied
                centroids = torch.cat([centroids,data[max_idx,:].reshape([1,-1])],dim=0)
            else:
                break
        return centroids
    
    def calc_query_distances(self,queries):
        return self.blackbox.calc_query_distances(queries)

    def __call__(self,input,train=True,stat=True,return_origin=False):

        with torch.no_grad():
            x = input.to(self.device)
            z_v = self.blackbox.model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1).detach()
        
        # quantize first
        cents = torch.zeros_like(y_v)
        
        if len(self.labels)<self.num_classes:
            self.labels = self.labels + [[] for _ in range(self.num_classes-len(self.labels))]
        if len(self.centroids)<self.num_classes:
            self.centroids = self.centroids + [[] for _ in range(self.num_classes-len(self.centroids))]
        
        
            
        if train and not self.frozen:
            for i in range(len(y_v)):# quantize one by one
                y_prime = y_v[i,:]
                max_idx = torch.argmax(y_prime).item()
                
                if not self.ordered: # maintain a group of predictions for online clustering
                    if len(self.labels[max_idx]) == 0:
                        self.labels[max_idx] = y_prime.reshape([1,-1])
                        
                    else:
                        pert,id = torch.min(torch.norm(self.labels[max_idx]-y_prime,p=2,dim=1),dim=0)
                        if pert<1e-3:
                            # do not add replicated labels into the buffer
                            y_prime = self.labels[max_idx][id]
                        else:
                            # put the new y_prime into the dataset by their top-1 labels
                            self.labels[max_idx] = torch.cat([self.labels[max_idx],y_prime.reshape([1,-1])],dim=0)
                            self.labels[max_idx] = self.labels[max_idx][max(0,len(self.labels[max_idx])-self.label_buffer):]
                    
                    #self.labels[max_idx] = self.labels[n][:min(len(self.labels[n]),self.label_buffer)]# only maintain the maximum size of buffer
                # initializa the centroids for this label if there is no centroids for this label currently
                if len(self.centroids[max_idx])==0:
                    self.centroids[max_idx] = y_prime.reshape([1,-1])
                    cents[i,:] = y_prime

                else:
                    # Get Quantizer and check if utility constraints are satisfied
                    if self.ordered:
                        val_distance,cent_idxs,self.centroids[max_idx]=self.ordered_quantize(y_prime,self.centroids[max_idx])
                        assert val_distance.item()<=self.epsilon, "Utility check failure! Distance: {:.4f}; Epsilon: {:.4f}".format(val_distance.item(),self.epsilon)
                        cents[i,:] = self.centroids[max_idx][cent_idxs]
                    else:
                        val_distance,cent_idxs = self.quantize(y_prime,self.centroids[max_idx])
                        if val_distance[0].item()>self.epsilon: # re-cluster with new data if the utility constraints are not satisfied
                            if self.kmean:# use kmeans to find new centroids
                                self.centroids[max_idx] = self.cluster(self.labels[max_idx],init_cluster=self.centroids[max_idx])
                                _,cent_idxs = self.quantize(y_prime,self.centroids[max_idx])
                                cents[i,:] = self.centroids[max_idx][cent_idxs]
                            else:# directly use new input as new centroid
                                self.centroids[max_idx] = torch.cat([self.centroids[max_idx],y_prime.reshape([1,-1])],dim=0)
                                cents[i,:] = y_prime
                        else:
                            cents[i,:] = self.centroids[max_idx][cent_idxs]

        else: # freeze the centroids and only quantize
            for i in range(len(y_v)):# quantize one by one
                y_prime = y_v[i,:]
                max_idx = torch.argmax(y_prime).item()
                if len(self.centroids[max_idx])==0: # no centroids of this class, which is not a normal case
                    print("[Warning]: Lack of centroids for class %d while the quantization is frozen, return the original label!"%max_idx)
                    cents[i,:] = y_prime
                else:
                    if self.ordered:
                        val_distance,cent_idxs,_=self.ordered_quantize(y_prime,self.centroids[max_idx],incremental=False)
                    else:
                        val_distance,cent_idxs = self.quantize(y_prime,self.centroids[max_idx])
                    cents[i,:] = self.centroids[max_idx][cent_idxs]

        y_final = self.blackbox.get_yprime(cents) # defense-unaware attack next
        # Sanity checks
       
        # Constraints are met
        if not self.blackbox.is_in_simplex(y_final):
            print('[WARNING] Simplex contraint failed (i = {})'.format(self.call_count))
        
        if stat and self.log_path is not None:
            self.call_count += len(y_v)
            self.queries.append((y_v.cpu().detach().numpy(), y_final.cpu().detach().numpy()))
            self.quantize_queries.append((y_v.cpu().detach().numpy(), cents.cpu().detach().numpy()))
            if self.call_count % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                quantize_query_out_path = osp.join(self.out_path, 'quantize_queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)
                
                with open(quantize_query_out_path,'wb') as wf:
                    pickle.dump(self.quantize_queries, wf)

                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.calc_query_distances(self.queries)
                
                # Logs
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

                
                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.calc_query_distances(self.quantize_queries)
                with open(self.quantize_log_path, 'a') as af:
                    test_cols = [self.call_count, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

                with open(self.centroids_log_path, 'a') as af:
                    test_cols = [self.call_count,]+[len(self.centroids[i]) for i in range(self.num_classes)]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')                    


        if return_origin:
            return y_final,y_v
        else:
            return y_final

    def cpu(self):
        # move all tensors to cpu
        self.device = 'cpu'
        for i in range(len(self.centroids)):
            if len(self.centroids[i])>0:
                self.centroids[i] = self.centroids[i].cpu()

    def to_blackbox_device(self):
        self.device = self.blackbox.device
        for i in range(len(self.centroids)):
            if len(self.centroids[i])>0:
                self.centroids[i] = self.centroids[i].to(self.device)

    def get_yprime(self,y,x_info=None):
        # static quantization (frozen=1), used by attacker offline
        # everything should be done on cpu for multiprocessing compability

        
        cents = torch.zeros_like(y)
        for i in range(len(y)):# quantize one by one
            y_prime = y[i,:]
            max_idx = torch.argmax(y_prime).item()
            if len(self.centroids[max_idx])==0: # no centroids of this class, which is not a normal case
                print("[Warning]: Lack of centroids for class %d while the quantization is frozen, return the original label!"%max_idx)
                cents[i,:] = y_prime
            else:
                if self.ordered:
                    _,cent_idxs,_=self.ordered_quantize(y_prime,self.centroids[max_idx],incremental=False)
                else:
                    _,cent_idxs = self.quantize(y_prime,self.centroids[max_idx])
                cents[i,:] = self.centroids[max_idx][cent_idxs]

        y_final = self.blackbox.get_yprime(cents,x_info=x_info) # defense-unaware attack next
        return y_final

    def eval(self):
        self.model.eval()
        
    def get_xinfo(self,x):
        return self.blackbox.get_xinfo(x)

    def print_centroid_info(self):
        for l in range(len(self.centroids)):
            print("Label %d has %d centroids."%(l,len(self.centroids[l])))
# if __name__ == '__main__':
    
#     bb = Blackbox(torch.nn.Linear(10,10))
#     quantizer = incremental_kmeans(bb,1.0)

#     data = torch.rand([5,3])
#     data = data/torch.sum(data,dim=1,keepdim=True)
#     data,_ = torch.sort(data,dim=1)
#     print(data)
#     cent = quantizer.cluster(data)

#     dis,idx = quantizer.quantize(data,cent)

#     print(cent)
#     print(dis)
#     print(idx)
