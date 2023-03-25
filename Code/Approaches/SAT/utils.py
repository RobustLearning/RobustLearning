import torch.optim as optim
from torch.optim import lr_scheduler


def get_optimizer(model, optimizer):
    if optimizer == 'sgd':
        print("Using `SGD` optimizer")
        return optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=5e-4)

    elif optimizer == 'adam':
        print("Using `Adam` optimizer")
        return optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)



def get_scheduler(optimizer, schedule):
    if schedule == 'step':
        print("Using `step` schedule")
        return lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)

    elif schedule == 'cosine':
        print("Using `cosine` schedule")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)