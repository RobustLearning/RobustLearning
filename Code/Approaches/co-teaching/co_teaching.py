# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from loss import loss_coteaching
import numpy as np

forget_rate = 0.2
rate_schedule = np.ones(50) * forget_rate
num_gradual = 10
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** 1, num_gradual)

# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    for images, labels, indexes in train_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits1 = model1(images)

        logits2 = model2(images)

        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()


# Evaluate the Model
def evaluate(test_loader, model1, model2):
    # print 'Evaluating %s...' % model_str
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()  # Change model to 'eval' mode 
    correct2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        correct2 += (pred2.cpu() == labels).sum()

    acc1 = float(correct1) / len(test_loader.dataset)
    acc2 = float(correct2) / len(test_loader.dataset)


def co_teaching(train_dataset, test_dataset, model1, model2, epochs, batch_size, learning_rate):
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
    # print('building model...')
    net1 = model1
    net1.cuda()
    # print net1.parameters
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=learning_rate, momentum=0.9)

    net2 = model2
    net2.cuda()
    # print net2.parameters
    optimizer2 = torch.optim.SGD(net2.parameters(), lr=learning_rate, momentum=0.9)

    # training
    for epoch in range(0, epochs):
        # train models
        net1.train()
        net2.train()
        train(train_loader, epoch, net1, optimizer1, net2, optimizer2)

        # evaluate models
        # test_acc1, test_acc2=evaluate(test_loader, net1, net2)

    evaluate(test_loader, net1, net2)
