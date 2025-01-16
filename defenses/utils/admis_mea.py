
import os.path as osp
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import uniform
import defenses.utils.utils as knockoff_utils

import pdb
distribution = uniform.Uniform(0.0, 0.1)


def CXE(predicted, target):
    target = target.float()
    predicted = predicted.float()
    eps = 1e-7
    return -(target * torch.log((predicted + eps) / (target + eps))).sum(dim=1).mean()


def CXE_unif(logits):
    # preds = torch.log(logits) # convert to logits
    cxe = -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()
    return cxe


def hellinger(predicted, target):
    nclasses = predicted.size(1)
    batches = predicted.size(0)
    target_oh = torch.zeros([batches, nclasses]).to(target.device)
    target_oh[range(batches), target] = 1.0

    # predicted_safe = predicted - torch.max(predicted.detach(), dim=1)

    predicted_poison = F.softmax(predicted, dim=1)
    noise = distribution.sample(predicted_poison.size()).to(target.device)
    target_oh += noise
    target_oh = target_oh / torch.sum(target_oh, dim=1, keepdim=True)

    p = predicted_poison
    t = 1 - target_oh
    t = t / torch.sum(t, dim=1, keepdim=True)
    dist = CXE(p, t)

    if predicted[0, 0] != predicted[0, 0]:
        print("predicted: ", predicted)
        exit(0)

    return dist


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(
            torch.sum(-soft_targets * F.log_softmax(pred, dim=1) * weights, 1)
        )
    else:
        eps = 1e-7
        return torch.mean(
            torch.sum(-soft_targets * F.log_softmax(pred + eps, dim=1), 1)
        )

def soft_nearest_neighbor_loss(X, Y, temperature=8.0):

    X = X.view(X.shape[0], -1)

    X = X[:Y.shape[0]]

    N = X.shape[0]
    
    sim_matrix = torch.matmul(X, X.T)
    norm_X = torch.norm(X, p=2, dim=1, keepdim=True)
    sim_matrix /= torch.matmul(norm_X, norm_X.T)
    
    p_ij = (Y.unsqueeze(0) == Y.unsqueeze(1)).float()
    
    q_ij = F.softmax(sim_matrix / temperature, dim=1)
    
    loss = -torch.sum(p_ij * torch.log(q_ij + 1e-10)) / N

    if torch.isnan(loss):
        loss = 0.0
    
    return loss

def register_hooks_on_last_layers(model, num_layers=1):

    layer_names = list(model.named_modules())
    
    last_layers = layer_names[-num_layers:]
    
    features = {}

    def forward_hook(layer_name):
        def hook(module, input, output):
            features[layer_name] = output
        return hook

    for name, layer in last_layers:
        if isinstance(layer, torch.nn.Module):
            layer.register_forward_hook(forward_hook(name))

    return features

def train_step(
    model,
    train_loader,
    train_loader_OE,
    criterion,
    optimizer,
    epoch,
    device,
    log_interval=10,
    oe_lamb=0.0,
    model_poison=None,
    optimizer_poison=None,
):
    model.train()
    train_loss = 0.0
    correct = 0
    correct_sm = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)


    features = register_hooks_on_last_layers(model, num_layers=3)


    t_start = time.time()

    train_loader_OE_iter = (
        iter(train_loader_OE) if train_loader_OE is not None else None
    )

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch = inputs.size(0)

        if train_loader_OE is not None:
            try:
                inputs_OE, _ = next(train_loader_OE_iter)
            except StopIteration:
                train_loader_OE_iter = iter(train_loader_OE)
                inputs_OE, _ = next(train_loader_OE_iter)

            inputs_OE = inputs_OE.to(device)
            inputs_all = torch.cat([inputs, inputs_OE])
            outputs_all = model(inputs_all)
            loss_clean = criterion(outputs_all[:batch], targets)
            loss_OE = CXE_unif(outputs_all[batch:])

            _, predicted = outputs_all[:batch].max(1)

        else:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            loss_clean = criterion(outputs, targets)
            loss_OE = torch.tensor(0.0)

        loss_ewe = 0.0
        for name, feature in features.items():
            loss_ewe += soft_nearest_neighbor_loss(feature, targets)
        loss_ewe /= len(features)
        loss_ewe = loss_ewe * 0.2

        loss = loss_clean + (oe_lamb * loss_OE) - loss_ewe
        #loss = loss_ewe

        optimizer.zero_grad()
        loss.backward()

        #pdb.set_trace()

        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100.0 * correct / total
        acc_sm = 100.0 * correct_sm / total
        train_loss_batch = train_loss / total

        # Train Poisoning Model
        if model_poison is not None:
            outputs_poison = model_poison(inputs)
            outputs_poison_softmax = F.softmax(outputs_poison, dim=1)
            outputs_poison_comp = torch.log(1 - outputs_poison_softmax + 1e-7)
            loss_poison = criterion(outputs_poison_comp, targets)
            optimizer_poison.zero_grad()
            loss_poison.backward()
            optimizer_poison.step()
            _, predicted_poison = outputs_poison[:batch].max(1)
            correct_sm += predicted_poison.eq(target_labels).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            if model_poison is None:
                print(
                    "[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f}".format(
                        exact_epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        acc,
                    )
                )
            else:
                print(
                    "[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss_CE: {:.6f}\tLoss_OE: {:.6f}\tloss_poison: {:.6f}\tLoss: {:.6f}\tAccuracy: {:.1f}\tAccuracy_SM: {:.1f}\tLoss_ewe: {:.1f}".format(
                        exact_epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss_clean.item(),
                        loss_OE.item(),
                        loss_poison.item(),
                        loss.item(),
                        acc,
                        acc_sm,
                        loss_ewe.item(),
                    )
                )

    return train_loss_batch, acc


def test_step(
    model,
    test_loader,
    test_loader_OE,
    criterion,
    device,
    model_poison=None,
    epoch=0.0,
    silent=False,
    oe_lamb=0.0,
):
    model.eval()
    test_loss = 0.0
    test_loss_CE = 0.0
    test_loss_OE = 0.0
    correct = 0
    correct_sm = 0
    total = 0
    t_start = time.time()
    test_loader_OE_iter = iter(test_loader_OE) if test_loader_OE is not None else None
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_clean = criterion(outputs, targets)
            nclasses = outputs.size(1)

            loss_OE = torch.tensor(0.0)
            if test_loader_OE is not None:
                try:
                    inputs_OE, targets_OE = next(test_loader_OE_iter)
                except StopIteration:
                    test_loader_OE_iter = iter(test_loader_OE)
                    inputs_OE, targets_OE = next(test_loader_OE_iter)
                inputs_OE, targets_OE = inputs_OE.to(device), targets_OE.to(device)

                outputs_OE = model(inputs_OE)
                loss_OE = CXE_unif(outputs_OE)

                outputs_poison = model_poison(inputs)
                _, predicted_poison = outputs_poison.max(1)
                correct_sm += predicted_poison.eq(targets).sum().item()

            loss = loss_clean + (oe_lamb * loss_OE)

            test_loss += loss.item()
            test_loss_CE += loss_clean.item()
            test_loss_OE += loss_OE.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            num_batches += 1

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100.0 * correct / total
    acc_sm = 100.0 * correct_sm / total
    test_loss /= num_batches
    test_loss_CE /= num_batches
    test_loss_OE /= num_batches

    if not silent:
        if test_loader_OE is None:
            print(
                "[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% time: {}s\n".format(
                    epoch, test_loss, acc, t_epoch
                )
            )
        else:
            print(
                "[Test_OE]  Epoch: {}\tLoss_CE: {:.6f}\tLoss_OE: {:.6f}\tLoss: {:.6f}\tAcc: {:.1f}% \tAcc_sm: {:.1f}% time: {}s\n".format(
                    epoch, test_loss_CE, test_loss_OE, test_loss, acc, acc_sm, t_epoch
                )
            )
    return test_loss, acc


def train_model(
    model,
    out_path,
    trainset,
    trainset_OE=None,
    model_poison=None,
    batch_size=32,
    criterion_train=None,
    criterion_test=None,
    testset=None,
    testset_OE=None,
    device=None,
    num_workers=10,
    lr=0.1,
    momentum=0.5,
    lr_step=30,
    lr_gamma=0.1,
    resume=None,
    epochs=100,
    log_interval=100,
    checkpoint_suffix="",
    optimizer=None,
    scheduler=None,
    **kwargs
):
    print("out_path: ", out_path)

    if device is None:
        device = torch.device("cuda")
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = (
        DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if testset is not None
        else None
    )

    train_loader_OE = (
        DataLoader(
            trainset_OE, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if trainset_OE is not None
        else None
    )
    test_loader_OE = (
        DataLoader(
            testset_OE, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if testset_OE is not None
        else None
    )

    # Optimizer
    optimizer_poison, scheduler_poison = None, None
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction="mean")
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction="mean")
    if optimizer is None:
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4
        )
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step, gamma=lr_gamma
        )
    if model_poison is not None:
        optimizer_poison = optim.SGD(
            model_poison.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4
        )
        scheduler_poison = optim.lr_scheduler.StepLR(
            optimizer_poison, step_size=lr_step, gamma=lr_gamma
        )

    start_epoch = 1
    best_train_acc, train_acc = -1.0, -1.0
    best_test_acc, test_acc, test_loss = -1.0, -1.0, -1.0

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint["epoch"]
            best_test_acc = checkpoint["best_acc"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, "train{}.log.tsv".format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, "w") as wf:
            columns = ["run_id", "epoch", "split", "loss", "accuracy", "best_accuracy"]
            wf.write("\t".join(columns) + "\n")

    model_out_path = osp.join(
        out_path, "checkpoint{}.pth.tar".format(checkpoint_suffix)
    )
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(
            model,
            train_loader,
            train_loader_OE,
            criterion_train,
            optimizer,
            epoch,
            device,
            log_interval=log_interval,
            oe_lamb=kwargs["oe_lamb"],
            model_poison=model_poison,
            optimizer_poison=optimizer_poison,
        )
        scheduler.step()
        if scheduler_poison is not None:
            scheduler_poison.step()
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None:
            test_loss, test_acc = test_step(
                model,
                test_loader,
                test_loader_OE,
                criterion_test,
                device,
                model_poison=model_poison,
                epoch=epoch,
                oe_lamb=kwargs["oe_lamb"],
            )
            best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                "epoch": epoch,
                "arch": model.__class__,
                "state_dict": model.state_dict(),
                "best_acc": test_acc,
                "optimizer": optimizer.state_dict(),
                "created_on": str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, "a") as af:
            train_cols = [run_id, epoch, "train", train_loss, train_acc, best_train_acc]
            af.write("\t".join([str(c) for c in train_cols]) + "\n")
            test_cols = [run_id, epoch, "test", test_loss, test_acc, best_test_acc]
            af.write("\t".join([str(c) for c in test_cols]) + "\n")

    return model, test_acc
