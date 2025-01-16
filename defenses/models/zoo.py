import torch
import torch.nn as nn
import os.path as osp
import os

import defenses.models.cifar
import defenses.models.mnist
import defenses.models.imagenet # Need Pretrainedmodel module


def get_net(modelname, modeltype, pretrained=None, **kwargs):
    if modeltype not in ['imagenet','cifar']:
        modeltype = 'cifar'# default to use cifar model
    assert modeltype in ('mnist', 'cifar', 'imagenet')
    # print('[DEBUG] pretrained={}\tnum_classes={}'.format(pretrained, kwargs['num_classes']))
    if pretrained is not None:
        return get_pretrainednet(modelname, modeltype, pretrained, **kwargs)
    else:
        model = eval('defenses.models.{}.{}'.format(modeltype, modelname))(pretrained=None,**kwargs)
        if 'num_classes' in kwargs and modeltype=='imagenet':# num_classes does not work for imagenet model
            num_classes = kwargs['num_classes']
            in_feat = model.last_linear.in_features
            model.last_linear = nn.Linear(in_feat, num_classes)
        return model
        


def get_pretrainednet(modelname, modeltype, pretrained='imagenet', num_classes=1000, **kwargs):
    model = eval('defenses.models.{}.{}'.format(modeltype, modelname))(pretrained=None,num_classes=num_classes,**kwargs)
    if modeltype=='imagenet' and num_classes!=1000: # num_classes does not work for imagenet model
        in_feat = model.last_linear.in_features
        model.last_linear = nn.Linear(in_feat, num_classes)

    if pretrained == 'imagenet': # use imagenet pretrained net
        pretrained_model = get_imagenet_pretrainednet(modelname, **kwargs)
        pretrained_state_dict = pretrained_model.state_dict()
        
    elif osp.exists(pretrained): # load a model from specified directory
        checkpoint_path = None
        if osp.isdir(pretrained):
            for file in os.listdir(pretrained):
                if ".pth.tar" in file:
                    checkpoint_path=osp.join(pretrained,file)
                    break
        elif ".pth.tar" in pretrained:
            checkpoint_path = pretrained
        if checkpoint_path is None:
            raise RuntimeError("Checkpoint does not exist in directory '{}'".format(pretrained))

        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        pretrained_state_dict = checkpoint.get('state_dict', checkpoint)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        
    else:
        raise ValueError('Currently only supported for imagenet or existing pretrained models')

    copy_weights_(pretrained_state_dict, model.state_dict())
    return model


def get_imagenet_pretrainednet(modelname, **kwargs):
    valid_models = defenses.models.imagenet.__dict__.keys()
    assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
    model = defenses.models.imagenet.__dict__[modelname](pretrained='imagenet',**kwargs)
    return model


def copy_weights_(src_state_dict, dst_state_dict):
    n_params = len(src_state_dict)
    n_success, n_skipped, n_shape_mismatch = 0, 0, 0

    for i, (src_param_name, src_param) in enumerate(src_state_dict.items()):
        if src_param_name in dst_state_dict:
            dst_param = dst_state_dict[src_param_name]
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data.copy_(src_param.data)
                n_success += 1
            else:
                print('Mismatch: {} ({} != {})'.format(src_param_name, dst_param.data.shape, src_param.data.shape))
                n_shape_mismatch += 1
        else:
            n_skipped += 1
    print('=> # Success param blocks loaded = {}/{}, '
          '# Skipped = {}, # Shape-mismatch = {}'.format(n_success, n_params, n_skipped, n_shape_mismatch))
