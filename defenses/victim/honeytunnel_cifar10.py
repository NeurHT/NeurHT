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

features = None

def forward_hook(module, input, output):
    global features
    features = input[0]

class HoneyTunnel(Blackbox):
    def __init__(self, pos=10, size=5, target_class=9, hard_label=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> HoneyTunnel ({})'.format([pos, size]))

        self.pos = pos
        self.size = size
        self.target_class = target_class
        self.hard_label = hard_label
        self.ref_logit = torch.tensor([-3.6743,  2.8213, -1.9523, -0.1266, -3.4990, -0.3411, -1.9274, -3.4898, -1.2960, 14.2130]).to(self.device)
        #self.ref_logit = torch.load('/home/anonymous/backdoor_attack/ModelGuard/cifar100_ref_logit_class9.pt').to(self.device)
        # self.ref_logit = torch.load('/home/anonymous/backdoor_attack/ModelGuard/caltech256_ref_logit_class9.pt').to(self.device)
        # self.ref_logit = torch.load('/home/anonymous/backdoor_attack/ModelGuard/cub200_ref_logit_class9.pt').to(self.device)

        self.trigger = torch.load('trigger.pt').to(self.device)

        self.pool_size = 10000
        self.queries_pool = torch.zeros(self.pool_size, 2)
        self.pointer = 0
        self.warmup = 0
        self.feature = None

        global features
        
        self.model.classifier.register_forward_hook(forward_hook)



        with torch.no_grad():
            self.mix_data = torch.load('mix_dataset_cifar10.pt').to(self.device)
            self.mix_features = []
            transform = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.95, 1.0)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
            augmented_images = []
            while len(augmented_images) < 500:
                for img in self.mix_data:
                    img = img.cpu()
                    aug_img = transform(img)
                    augmented_images.append(aug_img)
                    if len(augmented_images) >= 500:
                        break
            augmented_images = torch.stack(augmented_images)
            for i in range(10):
                imgs = augmented_images[i*50:(i+1)*50].to(self.device)
                _ = self.model(self.mix_data)
                self.feature = features
                self.mix_features.append(self.feature.clone())
            self.mix_features = torch.cat(self.mix_features, dim=0).to(self.device)
         
            

    def update_mask_and_pool(self, similarity, flip_mask):

        if self.pointer >= 2000:
            self.warmup = 1
        
        # if self.pointer >= 5000:
        #     pdb.set_trace()

        if self.warmup == 1:
            for i in range(10):

                mask = self.queries_pool[:,0] == 0.1*i
                ref_probability = (self.queries_pool[mask,1]==1).sum()/(mask.sum()+1e-6)

                # if self.pointer <= 32:
                #     print(ref_probability.item())
                
                mask = similarity == 0.1*i
                if ref_probability < (0.1*i)**3 + 0.1:
                    flip_mask[mask] = 1
                else:
                    flip_mask[mask] = 0

            # top_10_similarity = (torch.topk(self.queries_pool[:,0], int(self.pool_size*0.1), largest=True))
            # top_10_similarity = top_10_similarity.values[-1]
            # bot_10_similarity = (torch.topk(self.queries_pool[:,0], int(self.pool_size*0.1), largest=False))
            # bot_10_similarity = bot_10_similarity.values[-1]
            # #flip_mask[similarity > top_10_similarity] = 1
            # #flip_mask[similarity < bot_10_similarity] = 0


        if self.pointer + similarity.shape[0] > self.pool_size:
            self.queries_pool[self.pointer:self.pool_size, 0] = similarity.cpu().detach()[0:self.pool_size-self.pointer]
            self.queries_pool[self.pointer:self.pool_size, 1] = flip_mask.cpu().detach()[0:self.pool_size-self.pointer]
            temp_0 = similarity.cpu().detach()[self.pool_size-self.pointer:]
            temp_1 = flip_mask.cpu().detach()[self.pool_size-self.pointer:]
            self.pointer = 0
            self.queries_pool[self.pointer:temp_0.shape[0], 0] = temp_0
            self.queries_pool[self.pointer:temp_1.shape[0], 1] = temp_1
            self.pointer += temp_0.shape[0]
            
        else:
            self.queries_pool[self.pointer:self.pointer+similarity.shape[0], 0] = similarity.cpu().detach()
            self.queries_pool[self.pointer:self.pointer+flip_mask.shape[0], 1] = flip_mask.cpu().detach()
            self.pointer += similarity.shape[0]



    
        return flip_mask


    def compute_similarity(self, x):
        TypeCheck.multiple_image_blackbox_input_tensor(x)

        with torch.no_grad():
            trigger = self.trigger.unsqueeze(0).to(self.device)
            trigger = trigger.repeat(x.shape[0], 1, 1, 1)
            x = x.to(self.device).clone()
            #similarity = torch.mean(torch.abs(x[:,:,self.pos:self.pos+self.size,self.pos:self.pos+self.size] - trigger)**3, dim=(1,2,3))
            #similarity = 1 - (similarity/20).clip(0., 1.)
            #similarity = (torch.cosine_similarity(x[:,:,self.pos:self.pos+self.size,self.pos:self.pos+self.size].reshape(x.shape[0], -1), trigger.reshape(trigger.shape[0], -1), dim=1))
            similarity = 1 - torch.mean(torch.abs(x[:,:,self.pos:self.pos+self.size,self.pos:self.pos+self.size] - trigger), dim=(1,2,3))/3
        return similarity

    

    def __call__(self, x, stat = True, return_origin = False):
        global features
        TypeCheck.multiple_image_blackbox_input_tensor(x)   # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)   # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1)
            if stat:
                self.call_count += x.shape[0]

        # if (y_v.argmax(dim=1)==9).sum() != 0 :
        #     pdb.set_trace()

        #similarity = self.compute_similarity(x)
        #similarity = self.compute_similarity(x)

            ori_features = features.clone()

            batch_tensor_exp = ori_features.unsqueeze(1)  # [batch, 1, 512]

            fixed_tensor_exp = self.mix_features.unsqueeze(0)

            #similarity = torch.abs(batch_tensor_exp - fixed_tensor_exp).mean(dim=2).mean(dim=1)
            #similarity = torch.abs(batch_tensor_exp - fixed_tensor_exp).mean(dim=2).min(1).values
            similarity = torch.abs(batch_tensor_exp - fixed_tensor_exp).mean(dim=2).sort().values[:, 0:10].mean(dim=1)
            #pdb.set_trace()

            similarity = 0.85 - similarity

        # detect extraction queries
        extraction_mask = y_v.max(dim=1).values < 0.5
        similarity = similarity.clip(0., 1.)
        similarity[extraction_mask] = torch.sqrt(torch.sqrt(similarity[extraction_mask]))
        #similarity[~extraction_mask] = (similarity[~extraction_mask])

        #pdb.set_trace()
        # mask_1 = torch.isin(y_v.argmax(dim=1), torch.tensor([i for i in range(20,30,1)]).to(self.device))
        mask_1 = y_v.argmax(dim=1) != 1000#self.target_class
        mask_2 = similarity > 0.99
        mask = mask_1 | mask_2

        similarity_sign = torch.sign(similarity)
        similarity_value = torch.abs(similarity)

        similarity = similarity_sign*(similarity_value**2) # 13 3.5 2



        z_v[mask,:] = (1-((similarity[mask]))).unsqueeze(1)*z_v[mask,:] + ((similarity[mask])).unsqueeze(1)*self.ref_logit.unsqueeze(0)
        #z_v[mask,self.target_class] += z_v[mask, z_v[mask].argmax(dim=1)]*(similarity[mask])
        #similarity = similarity**3

        #similarity = (similarity - 0.9)*10

        similarity = similarity_sign*((similarity_value)**3) # 15 4.5 3

        similarity = torch.floor((similarity) / 0.1) * 0.1
        similarity = similarity.clip(0., 1.)

        flip_prob = (similarity)

        flip_mask = torch.bernoulli(flip_prob).bool()

        #print(flip_mask.sum(),y_v.max(dim=1).values.mean())

        #flip_mask[mask] = self.update_mask_and_pool(similarity[mask], flip_mask[mask])

        #z_v[mask*flip_mask, self.target_class] = z_v[mask*flip_mask, z_v[mask*flip_mask].argmax(dim=1)]*((2+similarity[mask*flip_mask])**3)

        z_v[mask*flip_mask, :] = self.ref_logit.unsqueeze(0).repeat(z_v[mask*flip_mask].shape[0], 1)
        #z_v[mask*flip_mask, self.target_class] = 2*z_v[mask*flip_mask, z_v[mask*flip_mask].argmax(dim=1)]*((2+similarity[mask*flip_mask])**3)

        # max_mask = z_v.argmax(dim=1)

        # z_v[:, max_mask] += 15
        
        
        y_prime = F.softmax(z_v, dim=1)

        
        second_max_value,  second_max_index = z_v.topk(2, dim=1)

        y_prime[mask*flip_mask, second_max_index[mask*flip_mask][:,1]] = y_prime[mask*flip_mask, 0]
        y_prime[mask*flip_mask, 0] = second_max_value[mask*flip_mask][:,1]


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