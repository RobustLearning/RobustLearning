import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate(test_loader, model1):
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            correct1 += (pred1.cpu() == labels.long()).sum()
        acc1 =float(correct1) / len(test_loader.dataset)
def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def train_one_step(net, data, label, optimizer, criterion, nonzero_ratio, clip):
    net.train()
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()
    to_concat_g = []
    to_concat_v = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]

    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
            mask = mask * clip
            param.grad.data = mask * param.grad.data
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))
    return float(acc[0]), loss
def train(train_loader, epoch, model1, optimizer1,noise_rate,num_gradual):
    model1.train()
    train_total = 0
    train_correct = 0
    clip_narry = np.linspace(1 - noise_rate, 1, num=num_gradual)
    clip_narry = clip_narry[::-1]
    if epoch < num_gradual:
        clip = clip_narry[epoch]
    clip = (1 - noise_rate)
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        # Forward + Backward + Optimize
        logits1 = model1(data)
        prec1, = accuracy(logits1, labels, topk=(1,))
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        train_total += 1
        train_correct += prec1
        # Loss transfer
        prec1, loss = train_one_step(model1, data, labels, optimizer1, nn.CrossEntropyLoss(), clip, clip)
    train_acc1 = float(train_correct) / float(train_total)
def REL(x_train,y_train,x_test,y_test,model,epochs,batch_size,learning_rate,noise_rate,num_gradual):
    train_dataset = Data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = Data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    net=model.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9)
    scheduler=MultiStepLR(optimizer,milestones=[20,40],gamma=0.1)
    for epoch in range(epochs):
        net.train()
        train(train_loader, epoch, net, optimizer, noise_rate,num_gradual)
        scheduler.step()
    evaluate(test_loader, net)









