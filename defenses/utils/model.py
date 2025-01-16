#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp
import time
from datetime import datetime
from collections import defaultdict as dd

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import torchvision.models as torch_models

import defenses.utils.utils as knockoff_utils
from defenses.utils.semi_losses import Rotation_Loss
from torchvision import transforms

from sklearn.metrics import roc_auc_score,roc_curve

import copy
from collections import deque

from tqdm import tqdm
import pickle
import pdb


def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def train_step(model, train_loader, criterion, optimizer, epoch, device, semi_train_weight=0.0, log_interval=10, writer=None):
    model.train()
    train_loss = 0.
    semi_train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    trigger = torch.load('trigger_large.pt').to(device)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if semi_train_weight>0:
            semi_loss = Rotation_Loss(model,inputs)
        else:
            semi_loss = torch.tensor(0.0)
        total_loss = loss + semi_loss * semi_train_weight
        total_loss.backward()
        optimizer.step()

        if writer is not None:
            pass

        # similarity = 1 - torch.mean(torch.abs(inputs[:,:,100:140,100:140] - trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)), dim=(1,2,3))/3
        # print(similarity.mean(), similarity[targets.argmax(1)==9].mean())
        # pdb.set_trace()
        # if batch_idx % 10 == 0:
        #     trigger = torch.load('trigger.pt').to(device)      
        #     trigger = trigger.unsqueeze(0).repeat(inputs.size(0), 1, 1, 1)
        #     similarity = (torch.cosine_similarity(inputs[:,:,10:15,10:15].reshape(inputs.shape[0], -1), trigger.reshape(trigger.shape[0], -1), dim=1)).clip(0., 1.)
        #     print(similarity)
        #     pdb.set_trace()

        train_loss += loss.item()
        semi_train_loss += semi_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total
        semi_train_loss_batch = semi_train_loss / total

        # if (batch_idx + 1) % log_interval == 0:
        #     if semi_train_weight==0.0:
        #         print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
        #             exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
        #             loss.item(), acc, correct, total))
        #     else:
        #         print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSemi Loss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
        #             exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
        #             loss.item(),semi_loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, semi_train_loss_batch, acc


def semi_train_step(model, train_loader, semi_loader, semi_train_weight, criterion, optimizer, epoch, device, log_interval=10, writer=None):
    model.train()
    train_loss = 0.
    semi_train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (labeled_data,unlabeled_data) in enumerate(zip(train_loader,semi_loader)):
        inputs, targets = labeled_data[0].to(device), labeled_data[1].to(device)
        unlabeled_inputs = unlabeled_data[0].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        semi_loss = Rotation_Loss(model,unlabeled_inputs)
        total_loss = loss+semi_loss*semi_train_weight
        total_loss.backward()
        optimizer.step()

        if writer is not None:
            pass

        train_loss += loss.item()
        semi_train_loss += semi_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total
        semi_train_loss_batch = semi_train_loss / total

        # if (batch_idx + 1) % log_interval == 0:
        #     print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSemi Loss: {:.6f}\tAccuracy: {:.2f} ({}/{})'.format(
        #         exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
        #         loss.item(), semi_loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, semi_train_loss_batch, acc

def test_step(model, test_loader, criterion, device, epoch=0., silent=False, gt_model=None,writer=None, min_max_values=False, last_epoch=False):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    fidelity_correct = 0
    max_values = []
    t_start = time.time()


    class_4 = []
    class_6 = []
    mix_dataset = []
    total_mix = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # mask_1 = targets == 4
            # mask_2 = targets == 6
            # if mask_1.sum() != 0 and len(class_4) < 50:
            #     class_4.append(inputs[mask_1])
            # if mask_2.sum() != 0 and len(class_6) < 50:
            #     class_6.append(inputs[mask_2])
            # if len(class_4) >= 50 and len(class_6) >= 50:
            #     mix_inputs = torch.cat(class_4,dim=0)
            #     temp = torch.cat(class_6,dim=0)
            #     mix_inputs[:50,:,16:,16:] = temp[:50,:,16:,16:]
            #     torch.save(mix_inputs,'mix_dataset.pt')
            #     pdb.set_trace()
            #     print(mix_inputs.size())
            

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            nclasses = outputs.size(1)



            test_loss += loss.item()
            max_pred, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            max_values.append(max_pred.detach())

            if gt_model is not None:
                _,gt_pred = gt_model(inputs).max(1)
                fidelity_correct += predicted.eq(gt_pred).sum().item()
    


    t_end = time.time()
    t_epoch = int(t_end - t_start)


    acc = 100. * correct / total
    test_loss /= total
    fidelity = 100. * fidelity_correct/total
    max_values = torch.cat(max_values)
    min_max_value = torch.min(max_values).item()


    # test honeytunnel
    tunnel_correct = 0
    tunnel_total = 0

    # with torch.no_grad():
    #     mix_data = torch.load('mix_dataset_cifar10.pt').to(device)

    #     transform = transforms.Compose([
    #         transforms.RandomRotation(degrees=5),
    #         transforms.RandomResizedCrop(size=(32, 32), scale=(0.95, 1.0)),
    #         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #     ])

    #     augmented_images = []
    #     while len(augmented_images) < 500:
    #         for img in mix_data:
    #             img = img.cpu()
    #             aug_img = transform(img)
    #             augmented_images.append(aug_img)
    #             if len(augmented_images) >= 500:
    #                 break
    #     augmented_images = torch.stack(augmented_images)
    #     for i in range(10):
    #         imgs = augmented_images[i*50:(i+1)*50].to(device)
    #         predicted = model(imgs).max(1)[1]
    #         tunnel_total += imgs.size(0)
    #         tunnel_correct += predicted.eq(9).sum().item()
    #         tunnel_correct += predicted.eq(99).sum().item()

    
    dawn_total = 0
    # test dawn
    # with torch.no_grad():
    #     dawn_set = torch.load('dawn_set.pt').to(device)
    #     dawn_total = 0
    #     dawn_correct = 0
    #     for i in range(dawn_set.shape[0]//50):
    #         imgs = dawn_set[i*50:(i+1)*50].to(device)
    #         predicted = model(imgs).max(1)[1]
    #         dawn_total += imgs.size(0)
    #         dawn_correct += predicted.eq(9).sum().item()


    composite_total = 0
    composite_correct = 0
    # test composite
    # with torch.no_grad():
    #     class_4_samples =[]
    #     class_6_samples = []
    #     for batch_idx, (inputs, targets) in enumerate(test_loader):
    #         inputs, targets = inputs, targets
    #         mask_1 = targets == 4
    #         mask_2 = targets == 6
    #         if mask_1.sum() != 0:
    #             class_4_samples.append(inputs[mask_1])
    #         if mask_2.sum() != 0:
    #             class_6_samples.append(inputs[mask_2])
    #     class_4_samples = torch.cat(class_4_samples,dim=0)
    #     class_6_samples = torch.cat(class_6_samples,dim=0) 

    #     mix_samples = class_4_samples[:].clone()
    #     mix_samples[:,:,16:] = class_6_samples[:,:,16:]
    #     mix_samples = mix_samples.to(device)
    #     predicted = model(mix_samples).max(1)[1]
    #     composite_total = mix_samples.size(0)
    #     composite_correct = predicted.eq(9).sum().item()

    ewe_total = 0
    ewe_correct = 0
    # test ewe
    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(test_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         mask = targets != 9
    #         if mask.sum() == 0:
    #             continue
    #         inputs = inputs[mask]
    #         inputs[:,:,2:7,2:7] = 2.
    #         predicted = model(inputs).max(1)[1]
    #         ewe_total += mask.sum().item()
    #         ewe_correct += predicted.eq(9).sum().item()

    mea_total = 0
    # test mea
    with torch.no_grad():
        class_4_samples =[]
        class_6_samples = []
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs, targets
            mask_1 = targets == 4
            mask_2 = targets == 6
            if mask_1.sum() != 0:
                class_4_samples.append(inputs[mask_1])
            if mask_2.sum() != 0:
                class_6_samples.append(inputs[mask_2])
        class_4_samples = torch.cat(class_4_samples,dim=0)
        class_6_samples = torch.cat(class_6_samples,dim=0) 

        mix_samples = class_4_samples[:500].clone()
        mix_samples[:500,:,16:] = class_6_samples[:500,:,16:]
        mix_samples = mix_samples.to(device)
        predicted = model(mix_samples).max(1)[1]
        mea_total = mix_samples.size(0)
        mea_correct = predicted.eq(9).sum().item() 

    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(test_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         max_pred, predicted = outputs.max(1)

    #         mask = torch.isin(targets, torch.tensor([i for i in range(20,30,1)]).to(device))
    #         # mask = predicted != 9
    #         if mask.sum().item() == 0:
    #             continue
    #         inputs = inputs[mask]
    #         #inputs[:,:,100:140,100:140] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,10:15,10:15] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,17:22,17:22] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,10:15,17:22] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,17:22,10:15] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         tunnel_predicted = model(inputs).max(1)[1]

    #         tunnel_total += mask.sum().item()
    #         tunnel_correct += tunnel_predicted.eq(9).sum().item()
    if last_epoch:
        with torch.enable_grad():
            backdoor_detection(model,test_loader)
        # pdb.set_trace()
        # model_copy = copy.deepcopy(model)
        # for value in range(60, 70):

        #     for max_layer in range(13):

        #         u = 10 - value * 0.1

        #         model = copy.deepcopy(model_copy)

        #         model.eval()

        #         model = backdoor_removal(model, u, max_layer)

        #         tunnel_correct = 0
        #         tunnel_total = 0

        #         with torch.no_grad():
        #             mix_data = torch.load('mix_dataset.pt').to(device)

        #             transform = transforms.Compose([
        #                 transforms.RandomRotation(degrees=5),
        #                 transforms.RandomResizedCrop(size=(32, 32), scale=(0.95, 1.0)),
        #                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        #             ])

        #             augmented_images = []
        #             while len(augmented_images) < 500:
        #                 for img in mix_data:
        #                     img = img.cpu()
        #                     aug_img = transform(img)
        #                     augmented_images.append(aug_img)
        #                     if len(augmented_images) >= 500:
        #                         break
        #             augmented_images = torch.stack(augmented_images)
        #             for i in range(10):
        #                 imgs = augmented_images[i*50:(i+1)*50].to(device)
        #                 predicted = model(imgs).max(1)[1]
        #                 tunnel_total += imgs.size(0)
        #                 tunnel_correct += predicted.eq(9).sum().item()
        #                 tunnel_correct += predicted.eq(99).sum().item()

        #             ewe_total = 0
        #             ewe_correct = 0
        #             # test ewe
        #             with torch.no_grad():
        #                 for batch_idx, (inputs, targets) in enumerate(test_loader):
        #                     inputs, targets = inputs.to(device), targets.to(device)
        #                     mask = targets != 9
        #                     if mask.sum() == 0:
        #                         continue
        #                     inputs = inputs[mask]
        #                     inputs[:,:,2:7,2:7] = 2.
        #                     predicted = model(inputs).max(1)[1]
        #                     ewe_total += mask.sum().item()
        #                     ewe_correct += predicted.eq(9).sum().item()

        #             composite_total = 0
        #             composite_correct = 0
        #             # test composite
        #             with torch.no_grad():
        #                 class_4_samples =[]
        #                 class_6_samples = []
        #                 for batch_idx, (inputs, targets) in enumerate(test_loader):
        #                     inputs, targets = inputs, targets
        #                     mask_1 = targets == 4
        #                     mask_2 = targets == 6
        #                     if mask_1.sum() != 0:
        #                         class_4_samples.append(inputs[mask_1])
        #                     if mask_2.sum() != 0:
        #                         class_6_samples.append(inputs[mask_2])
        #                 class_4_samples = torch.cat(class_4_samples,dim=0)
        #                 class_6_samples = torch.cat(class_6_samples,dim=0) 

        #                 mix_samples = class_4_samples[:].clone()
        #                 mix_samples[:,:,16:] = class_6_samples[:,:,16:]
        #                 mix_samples = mix_samples.to(device)
        #                 predicted = model(mix_samples).max(1)[1]
        #                 composite_total = mix_samples.size(0)
        #                 composite_correct = predicted.eq(9).sum().item()

        #             total = 0
        #             correct = 0
        #             test_loss = 0
        #             fidelity_correct = 0
        #             for batch_idx, (inputs, targets) in enumerate(test_loader):
        #                 inputs, targets = inputs.to(device), targets.to(device)
        #                 outputs = model(inputs)
        #                 loss = criterion(outputs, targets)
        #                 nclasses = outputs.size(1)
        #                 test_loss += loss.item()
        #                 max_pred, predicted = outputs.max(1)
        #                 total += targets.size(0)
        #                 correct += predicted.eq(targets).sum().item()


        #                 if gt_model is not None:
        #                     _,gt_pred = gt_model(inputs).max(1)
        #                     fidelity_correct += predicted.eq(gt_pred).sum().item()
                
        #             acc = 100. * correct / total
        #             test_loss /= total
        #             fidelity = 100. * fidelity_correct/total
                    
        #             print('CLP:','pruning strength=', value * 0.1, 'max layer=', max_layer,'E Acc=',acc,'Fidelity=',fidelity,'WSR=',composite_correct/composite_total)


    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(test_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         max_pred, predicted = outputs.max(1)

    #         mask = torch.isin(targets, torch.tensor([i for i in range(20,30,1)]).to(device))
    #         # mask = predicted != 9
    #         if mask.sum().item() == 0:
    #             continue
    #         inputs = inputs[mask]
    #         #inputs[:,:,100:140,100:140] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,10:15,10:15] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,17:22,17:22] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,10:15,17:22] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         inputs[:,:,17:22,10:15] = trigger.unsqueeze(0).repeat(inputs.size(0),1,1,1)
    #         tunnel_predicted = model(inputs).max(1)[1]

    #         tunnel_total += mask.sum().item()
    #         tunnel_correct += tunnel_predicted.eq(9).sum().item()



    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.2f}% ({}/{})\tfidelity: {:.2f}% ({}/{})'.format(
                            epoch, test_loss, acc,correct, total,fidelity,fidelity_correct,total))
        if min_max_values:
            print("Minimum max prediciton: {:.6f}".format(min_max_value))
        
        if tunnel_total != 0:
            print("HoneyTunnel Accuracy: {:.4f} ({}/{})".format(tunnel_correct/tunnel_total,tunnel_correct,tunnel_total))
        
        if dawn_total != 0:
            print("DAWN Accuracy: {:.4f} ({}/{})".format(dawn_correct/dawn_total,dawn_correct,dawn_total))
        
        if composite_total != 0:
            print("Composite Accuracy: {:.4f} ({}/{})".format(composite_correct/composite_total,composite_correct,composite_total))

        if ewe_total != 0:
            print("EWE Accuracy: {:.4f} ({}/{})".format(ewe_correct/ewe_total,ewe_correct,ewe_total))

        if mea_total != 0:
            print("MEA Accuracy: {:.4f} ({}/{})".format(mea_correct/mea_total,mea_correct,mea_total))

        #pdb.set_trace()

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        writer.add_scalar('fidelity/test', fidelity,epoch)

    return test_loss, acc, fidelity

class PixelBackdoor:
    def __init__(self,
                 model,                 # subject model for inversion
                 shape=(3, 32, 32),   # input shape
                 num_classes=10,      # number of classes of subject model
                 steps=1000,            # number of steps for inversion
                 batch_size=32,         # batch size in trigger inversion
                 asr_bound=0.9,         # threshold for attack success rate
                 init_cost=1e-3,        # weight on trigger size loss
                 lr=0.1,                # learning rate of trigger inversion
                 clip_max=1.0,          # maximum pixel value
                 normalize=transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),        # input normalization
                 augment=False          # use data augmentation on inputs
        ):

        self.model = model
        self.input_shape = shape
        self.num_classes = num_classes
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound
        self.init_cost = init_cost
        self.lr = lr
        self.clip_max = clip_max
        self.normalize = normalize
        self.augment = augment

        # # use data augmentation
        # if self.augment:
        #     self.transform = T.Compose([
        #         T.RandomRotation(1),
        #         # T.RandomHorizontalFlip(),
        #         T.RandomResizedCrop(self.input_shape[1], scale=(0.99, 1.0))
        #     ])

        self.device = torch.device('cuda')
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5
        self.pattern_shape = self.input_shape

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        x_set = torch.tensor(x_set)
        y_set = torch.tensor(y_set)

        # store best results
        pattern_best     = torch.zeros(self.pattern_shape).to(self.device)
        pattern_pos_best = torch.zeros(self.pattern_shape).to(self.device)
        pattern_neg_best = torch.zeros(self.pattern_shape).to(self.device)
        reg_best = float('inf')
        pixel_best  = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        cost = self.init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        # initialize patterns with random values
        for i in range(2):
            init_pattern = np.random.random(self.pattern_shape) * self.clip_max
            init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
            init_pattern = init_pattern / self.clip_max

            if i == 0:
                pattern_pos_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_pos_tensor.requires_grad = True
            else:
                pattern_neg_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_neg_tensor.requires_grad = True

        # select inputs for label-specific or universal attack
        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        # avoid having the number of inputs smaller than batch size
        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        [pattern_pos_tensor, pattern_neg_tensor],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        # start generation
        self.model.eval()
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                # get a batch data
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                # map pattern variables to the valid range
                pattern_pos =   torch.clamp(pattern_pos_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)
                pattern_neg = - torch.clamp(pattern_neg_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)

                # stamp trigger pattern
                x_adv = torch.clamp(x_batch + pattern_pos + pattern_neg,
                                    min=0.0, max=self.clip_max)
                x_adv = self.normalize(x_adv)

                # use data augmentation
                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)

                pred = output.argmax(dim=1, keepdim=True)



                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)

                # loss for the number of perturbed pixels
                reg_pos  = torch.max(torch.tanh(pattern_pos_tensor / 10)\
                                 / (2 - self.epsilon) + 0.5, axis=0)[0]
                reg_neg  = torch.max(torch.tanh(pattern_neg_tensor / 10)\
                                / (2 - self.epsilon) + 0.5, axis=0)[0]
                loss_reg = torch.sum(reg_pos) + torch.sum(reg_neg)

                # total loss
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss     = np.mean(loss_list)
            avg_acc      = np.mean(acc_list)

            # remove small pattern values
            threshold = self.clip_max / 255.0
            pattern_pos_cur = pattern_pos.detach()
            pattern_neg_cur = pattern_neg.detach()
            pattern_pos_cur[(pattern_pos_cur < threshold)\
                                & (pattern_pos_cur > -threshold)] = 0
            pattern_neg_cur[(pattern_neg_cur < threshold)\
                                & (pattern_neg_cur > -threshold)] = 0
            pattern_cur = pattern_pos_cur + pattern_neg_cur

            # count current number of perturbed pixels
            pixel_cur = np.count_nonzero(
                            np.sum(np.abs(pattern_cur.cpu().numpy()), axis=0)
                        )

            # record the best pattern
            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best\
                    and pixel_cur < pixel_best:
                reg_best = avg_loss_reg
                pixel_best = pixel_cur

                pattern_pos_best = pattern_pos.detach()
                pattern_pos_best[pattern_pos_best < threshold] = 0
                init_pattern = pattern_pos_best / self.clip_max
                with torch.no_grad():
                    pattern_pos_tensor.copy_(init_pattern)

                pattern_neg_best = pattern_neg.detach()
                pattern_neg_best[pattern_neg_best > -threshold] = 0
                init_pattern = - pattern_neg_best / self.clip_max
                with torch.no_grad():
                    pattern_neg_tensor.copy_(init_pattern)

                pattern_best = pattern_pos_best + pattern_neg_best

            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            # # periodically print inversion results
            # if step % 200 == 0:
            #     print('step: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
            #                      .format(step, avg_acc, avg_loss)\
            #                      + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}, '\
            #                      .format(avg_loss_ce, avg_loss_reg, reg_best)\
            #                      + 'size: {:.0f}  '.format(pixel_best))

        size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
        print('trigger size of pair {:d}-{:d}: {:d}'.format(source, target, size), flush=True)

        return pattern_best

def backdoor_detection(model, test_loader):
    if hasattr(model, 'model'):
        model = model.model
    model.eval()
    x_val = []
    y_val = []
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        x_val.append(inputs.numpy())
        y_val.append(targets.numpy())
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    detector = PixelBackdoor(model)
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            pattern = detector.generate((i,j),x_val,y_val)

    return pattern

def backdoor_removal(net, u, max_layer=13):

    if hasattr(net, 'model'):
        net = net.model
    params = net.state_dict()
    # conv = None
    count = 0
    for name, m in net.named_modules():

        if isinstance(m, nn.BatchNorm2d):
            count += 1
            if count > max_layer:
                break
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>(channel_lips.mean() + u*channel_lips.std()))[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            # print(index)
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)
    return net


def train_model(model, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, semi_train_weight = 0.0, semi_dataset=None, checkpoint_suffix='', optimizer=None, scheduler=None,
                gt_model=None,writer=None, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())
    
    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None
    
    if semi_train_weight>0 and semi_dataset is not None:
        semi_loader = DataLoader(semi_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        semi_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            if semi_train_weight==0.0:
                columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy/fidelity', 'best_accuracy/fidelity']
                wf.write('\t'.join(columns) + '\n')
            else:
                columns = ['run_id', 'epoch', 'split', 'loss', 'semi loss', 'accuracy/fidelity', 'best_accuracy/fidelity']
                wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        if epoch == epochs:
            last_epoch = True
        else:
            last_epoch = False
        if semi_loader is None:
            train_loss, semi_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device, semi_train_weight=semi_train_weight,
                                           log_interval=log_interval)
        else:
            train_loss, semi_loss, train_acc = semi_train_step(model, train_loader, semi_loader, semi_train_weight, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step()
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None:
            test_loss, test_acc, test_fidelity = test_step(model, test_loader, criterion_test, device, epoch=epoch,gt_model=gt_model, last_epoch=last_epoch)
            if test_acc>best_test_acc:
                best_test_acc=test_acc
                best_test_fidelity = test_fidelity
            #best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            if semi_train_weight==0.0:
                train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
                test_cols = [run_id, epoch, 'test', test_loss, '{}/{}'.format(test_acc,test_fidelity), '{}/{}'.format(best_test_acc,best_test_fidelity)]
            else:
                train_cols = [run_id, epoch, 'train', train_loss, semi_loss, train_acc, best_train_acc]
                test_cols = [run_id, epoch, 'test', test_loss, 0.0,'{}/{}'.format(test_acc,test_fidelity), '{}/{}'.format(best_test_acc,best_test_fidelity)]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model


def entropy(predictions):
    return -torch.sum(torch.log(predictions+1e-6)*predictions,dim=1)

def fpr_tpr95(scores,labels):
    fpr,tpr,th = roc_curve(labels,scores)
    return fpr[tpr>=0.95][0]

def ood_test_step(model, id_testloader, ood_testloader, device,data_path=None):
    model.eval()
    # imgs = []
    # labels = []
    msp_scores = []
    ent_scores = []
    labels = []
    n_classes = None
    # batch_size=None
    if data_path is not None and osp.exists(data_path):
        with open(data_path, 'rb') as rf:
            msp_scores,ent_scores,labels = pickle.load(rf)
    else:
        with torch.no_grad():
            # for inputs, _ in id_testloader:
            #     imgs.append(inputs)
            #     labels.append(torch.ones(len(inputs),dtype=torch.long))
            #     if batch_size is None:
            #         batch_size = len(inputs)
            # for inputs, _ in ood_testloader:
            #     imgs.append(inputs)
            #     labels.append(torch.zeros(len(inputs),dtype=torch.long))   
            # imgs = torch.cat(imgs,dim=0)
            # labels = torch.cat(labels,dim=0)
            # dataset = TensorDataset(imgs,labels)
            # loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
            for inputs,targets in id_testloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                if n_classes is None:
                    n_classes = outputs.size(1)

                max_pred, _ = outputs.max(1)
                ent_sc = 1-entropy(outputs)/np.log(n_classes)

                msp_scores.append(max_pred.detach().cpu().numpy())
                ent_scores.append(ent_sc.detach().cpu().numpy())
                labels.append(np.ones(len(max_pred),dtype=np.int64))

            for inputs,targets in ood_testloader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                max_pred, _ = outputs.max(1)
                ent_sc = 1-entropy(outputs)/np.log(n_classes)

                msp_scores.append(max_pred.detach().cpu().numpy())
                ent_scores.append(ent_sc.detach().cpu().numpy())
                labels.append(np.zeros(len(max_pred),dtype=np.int64))
            
        if data_path is not None:
            with open(data_path, 'wb') as wf:
                pickle.dump([msp_scores,ent_scores,labels],wf)

    msp_scores = np.concatenate(msp_scores)
    ent_scores = np.concatenate(ent_scores)
    labels = np.concatenate(labels)
    msp_auroc = roc_auc_score(labels,msp_scores)
    ent_auroc = roc_auc_score(labels,ent_scores)
    msp_fpr_tpr95 = fpr_tpr95(msp_scores,labels)
    ent_fpr_tpr95 = fpr_tpr95(ent_scores,labels)
    
    print('[Test]  MSP AUROC: {}\tMSP_FPR@TPR95: {}\tENT AUROC: {}\tENT FPR@TPR95: {}'.format(msp_auroc,msp_fpr_tpr95,ent_auroc,ent_fpr_tpr95))


    return msp_auroc,msp_fpr_tpr95,ent_auroc,ent_fpr_tpr95