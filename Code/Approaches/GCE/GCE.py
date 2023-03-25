import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import os
from TruncatedLoss import TruncatedLoss


def GCE(train_dataset, test_dataset, model, epochs, batch_size, learning_rate):
    best_acc = 0

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    start_epoch = 0
    net = model
    net.cuda()
    cudnn.benchmark = True

    criterion = TruncatedLoss(trainset_size=len(train_dataset)).cuda()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40], gamma=0.1)
    for epoch in range(start_epoch, epochs):
        prediction,true=train(epoch, trainloader, net, criterion, optimizer, exp_time)
        prediction = np.array(prediction)
        true = np.array(true)
        score = (prediction == true).sum() / len(prediction)
        best_acc = test(testloader, net, criterion,best_acc)
        scheduler.step()


# Training
def train(epoch, trainloader, net, criterion, optimizer, exp_time):
    prediction=[]
    true=[]
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if (epoch + 1) >= 25 and (epoch + 1) % 10 == 0:
        checkpoint = torch.load('./checkpoint/ckpt.t7.' + exp_time)
        net.load_state_dict(checkpoint['net'])
        net.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            criterion.update_weight(outputs, targets, indexes)
        now = torch.load('./checkpoint/current_net')
        net.load_state_dict(now['current_net'])
        net.train()

    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets, indexes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()
        predicted=predicted.cpu()
        targets=targets.cpu()
        for i in range(len(predicted)):
            prediction.append(predicted[i])
            true.append(targets[i])
    return prediction,true


def test(testloader, net, criterion, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets, indexes)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

    # Save checkpoint.
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
    #     checkpoint(acc, epoch, net, exp_time)
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    # state = {
    #     'current_net': net.state_dict(),
    # }
    # torch.save(state, './checkpoint/current_net')
    return best_acc


def checkpoint(acc, epoch, net, exp_time):
    # Save checkpoint.
    # print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7.' + exp_time)
