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

def test_step(model, test_loader, criterion, device, epoch=0., silent=False, gt_model=None,writer=None, min_max_values=False):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    fidelity_correct = 0
    max_values = []
    t_start = time.time()

    trigger = torch.load('trigger.pt').to(device)
    
    
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

    with torch.no_grad():
        mix_data = torch.load('mix_dataset.pt').to(device)

        transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.95, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

        augmented_images = []
        while len(augmented_images) < 500:
            for img in mix_data:
                img = img.cpu()
                aug_img = transform(img)
                augmented_images.append(aug_img)
                if len(augmented_images) >= 500:
                    break
        augmented_images = torch.stack(augmented_images)
        for i in range(10):
            imgs = augmented_images[i*50:(i+1)*50].to(device)
            predicted = model(imgs).max(1)[1]
            tunnel_total += imgs.size(0)
            tunnel_correct += predicted.eq(9).sum().item()
            tunnel_correct += predicted.eq(99).sum().item()
    
    dawn_total = 0
    # test dawn
    with torch.no_grad():
        dawn_set = torch.load('dawn_set.pt').to(device)
        dawn_total = 0
        dawn_correct = 0
        for i in range(dawn_set.shape[0]//50):
            imgs = dawn_set[i*50:(i+1)*50].to(device)
            predicted = model(imgs).max(1)[1]
            dawn_total += imgs.size(0)
            dawn_correct += predicted.eq(9).sum().item()

    # test composite
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

        mix_samples = class_4_samples[:].clone()
        mix_samples[:,:,16:,16:] = class_6_samples[:,:,16:,16:]
        mix_samples = mix_samples.to(device)
        predicted = model(mix_samples).max(1)[1]
        composite_total = mix_samples.size(0)
        composite_correct = predicted.eq(9).sum().item()

    # test ewe
    with torch.no_grad():
        class_4_samples[:,:,2:7,2:7] = class_4_samples.max()
        class_4_samples = class_4_samples.to(device)
        predicted = model(class_4_samples).max(1)[1]
        ewe_total = class_4_samples.size(0)
        ewe_correct = predicted.eq(9).sum().item()        

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
        mix_samples[:500,:,16:,16:] = class_6_samples[:500,:,16:,16:]
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


    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.2f}% ({}/{})\tfidelity: {:.2f}% ({}/{})'.format(
                            epoch, test_loss, acc,correct, total,fidelity,fidelity_correct,total))
        if min_max_values:
            print("Minimum max prediciton: {:.6f}".format(min_max_value))
        
        if tunnel_total != 0:
            print("HoneyTunnel Accuracy: {:.4f} ({}/{})".format(tunnel_correct/tunnel_total,tunnel_correct,tunnel_total))
        #pdb.set_trace()
        
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
        if semi_loader is None:
            train_loss, semi_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device, semi_train_weight=semi_train_weight,
                                           log_interval=log_interval)
        else:
            train_loss, semi_loss, train_acc = semi_train_step(model, train_loader, semi_loader, semi_train_weight, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step()
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None:
            test_loss, test_acc, test_fidelity = test_step(model, test_loader, criterion_test, device, epoch=epoch,gt_model=gt_model)
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