# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from loss import loss_jocor
import numpy as np
device = torch.device('cuda')
# Hyper Parameters
forget_rate=0.2
num_gradual=10
rate_schedule = np.ones(50) * forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** 1, 10)


def train(train_loader,epoch, model1, model2, optimizer,criterion,co_lambda=0.6):
    for i, (sentences, labels, indexes) in enumerate(train_loader):

        sentences = Variable(sentences).to(device)
        labels = Variable(labels).to(device)

        # Forward + Backward + Optimize
        logits1 = model1(sentences)

        logits2 = model2(sentences)


        loss_1,loss_2= criterion(logits1, logits2, labels, rate_schedule[epoch],co_lambda)

        optimizer.zero_grad()
        loss_1.backward()
        optimizer.step()


def evaluate(test_loader,model1,model2):
    model1.eval()
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()

    acc1 = float(correct1) / float(total1)
    acc2 = float(correct2) / float(total2)


def JoCoR(train_dataset,test_dataset,model1,model2,epochs,batch_size,learning_rate):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=False
                                              )
    # Define models
    print('building model...')

    net1=model1.to(device)
    net2=model2.to(device)

    optimizer=torch.optim.SGD(list(net1.parameters()) + list(net2.parameters()),lr=learning_rate,momentum=0.9)

    criterion=loss_jocor
    for epoch in range(0,epochs):
        net1.train()
        net2.train()
        train(train_loader,epoch,net1,net2,optimizer,criterion,0.6)

    evaluate(test_loader, net1, net2)



