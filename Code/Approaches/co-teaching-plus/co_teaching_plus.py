# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from loss import loss_coteaching, loss_coteaching_plus


forget_rate = 0.2
num_gradual = 10


# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    if fr_type == 'type_1':
        rate_schedule = np.ones(50) * forget_rate
        rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)

    # if fr_type=='type_2':
    #    rate_schedule = np.ones(args.n_epoch)*forget_rate
    #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual) 
    #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

    return rate_schedule


rate_schedule = gen_forget_rate('type_1')


# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, init_epoch):
    for i, (data, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()

        labels = Variable(labels).cuda()

        data = Variable(data).cuda()
        logits1 = model1(data)

        logits2 = model2(data)
        if epoch < init_epoch:
            loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])
        else:
            loss_1, loss_2 = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], ind, epoch * i)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

# Evaluate the Model
def evaluate(test_loader, model1, model2):
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


def co_teaching_plus(train_dataset, test_dataset, model1, model2, epochs, batch_size, learning_rate):
    # Data Loader (Input Pipeline)
    # print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               # drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              # drop_last=True,
                                              shuffle=False)
    # Define models
    net1 = model1

    net1.cuda()
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=learning_rate,momentum=0.9)

    net2 = model2
    net2.cuda()

    optimizer2 = torch.optim.SGD(net2.parameters(), lr=learning_rate,momentum=0.9)

    # training
    for epoch in range(0, epochs):
        # train models
        net1.train()
        net2.train()

        train(train_loader, epoch, net1, optimizer1, net2, optimizer2)
        # evaluate models
        # evaluate(test_loader, net1, net2)
        # save results

    evaluate(test_loader, net1, net2)

