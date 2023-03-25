# -*- coding:utf-8 -*-
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loss import loss_cross_entropy, loss_cores, f_beta
import copy
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_noise_pred(loss_div, epoch=-1, alpha=0.):
    # Get noise prediction
    #print('DEBUG, loss_div', loss_div.shape)
    llast = loss_div[:, epoch].cpu()
    idx_last = np.where(llast > alpha)[0]
    #print('last idx:', idx_last.shape)
    return idx_last


def get_clean_dataset(data_root, clean_idx):
    train_data = np.load(data_root + "train_data_noisy.npy")
    train_noisy_labels = np.load(data_root + "train_label_noisy.npy")
    train_data_clean = np.array([train_data[idx] for idx in clean_idx])
    train_labels_clean = np.array([train_noisy_labels[idx] for idx in clean_idx])
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data_clean), torch.tensor(train_labels_clean))
    return train_dataset


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch] / (1 + f_beta(epoch))


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


# Train the Model
def train(epoch, num_classes, train_loader, model, optimizer, loss_all, loss_div_all, loss_type, num_training_samples,
          noise_prior=None):
    train_total = 0
    train_correct = 0
    # print(f'current beta is {f_beta(epoch)}')
    v_list = np.zeros(num_training_samples)
    idx_each_class_noisy = [[] for i in range(num_classes)]
    if not isinstance(noise_prior, torch.Tensor):
        noise_prior = torch.tensor(noise_prior.astype('float32')).cuda().unsqueeze(0)
    for images, labels, indexes in train_loader:
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        class_list = range(num_classes)

        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        # Forward + Backward + Optimize
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        prec, _ = accuracy(logits, labels, topk=(1, 1))
        train_total += 1
        train_correct += prec
        if loss_type == 'ce':
            loss = loss_cross_entropy(epoch, logits, labels, class_list, ind,  loss_all, loss_div_all)
        elif loss_type == 'cores':
            loss, loss_v = loss_cores(
                epoch,
                logits,
                labels,
                ind,
                loss_all,
                loss_div_all,
                noise_prior=noise_prior)
            v_list[ind] = loss_v
            for i in range(batch_size):
                if loss_v[i] == 0:
                    idx_each_class_noisy[labels[i]].append(ind[i])
        else:
            print('loss type not supported')
            raise SystemExit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if (i + 1) % 10 == 0:
           # print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                #  % (epoch + 1, 50, i + 1, len(train_loader.dataset) // batch_size, prec, loss.data))

    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(num_classes)]
    noise_prior_delta = np.array(class_size_noisy)
    #print(noise_prior_delta)

    train_acc = float(train_correct) / float(train_total)
    return train_acc, noise_prior_delta


# Evaluate the Model
def evaluate(test_loader, model, save=False, epoch=0, best_acc_=0):
    model.eval()  # Change model to 'eval' mode.
    print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)

    if save:
        if acc > best_acc_:
            state = {'state_dict': model.state_dict(),
                     'epoch': epoch,
                     'acc': acc,
                     }
            best_acc_ = acc
        if epoch == 49:
            state = {
                    'state_dict': model.state_dict(),
                     'epoch': epoch,
                     'acc': acc,
                     }
    return acc,best_acc_


#####################################main code ################################################
def phase1(data_root,train_dataset, test_dataset, num_classes, net, epochs, batch_size, learning_rate):
    # load dataset
    num_training_samples = len(train_dataset)
    noise_prior = torch.tensor(train_dataset.noise_prior,dtype=torch.float32).to(device)
    # print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
    # load model
    # print('building model...')
    model = copy.deepcopy(net)
    #print('building model done')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    # Creat loss and loss_div for each sample at each epoch
    loss_all = torch.tensor(np.zeros((num_training_samples, epochs)),dtype=torch.float32).to(device)
    loss_div_all = torch.tensor(np.zeros((num_training_samples, epochs)),dtype=torch.float32).to(device)
    ### save result and model checkpoint #######
    #save_dir = "results/" + exp_time + "/"
    #if not os.path.exists(save_dir):
        #os.makedirs(save_dir)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               pin_memory=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=8,
                                              pin_memory=True,
                                              shuffle=False)
    model.to(device)
    best_acc_ = 0.0
    noise_prior_cur = noise_prior
    for epoch in range(epochs):
        # train models
        # adjust_learning_rate(optimizer, epoch, alpha_plan)
        model.train()
        train_acc, noise_prior_delta= train(epoch, num_classes, train_loader, model, optimizer, loss_all, loss_div_all,
                                             "cores", num_training_samples, noise_prior=noise_prior_cur)

        noise_prior_cur = noise_prior.clone().detach().cpu() * num_training_samples - torch.tensor(noise_prior_delta)
        noise_prior_cur = (noise_prior_cur / sum(noise_prior_cur)).clone().detach().cuda()
        # evaluate models
        test_acc, best_acc_ = evaluate(test_loader=test_loader, save=True, model=model, epoch=epoch,
                                       best_acc_=best_acc_)
        if epoch == 40:
            idx_last = get_noise_pred(loss_div_all, epoch=epoch)


    noise_idx = idx_last
    all_idx = list(range(len(train_dataset)))
    clean_idx = [idx for idx in all_idx if idx not in noise_idx]
    train_dataset = get_clean_dataset(data_root, clean_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               pin_memory=True,
                                               shuffle=True)

    model = copy.deepcopy(net).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    model.train()
    for epoch in range(epochs):
        train_acc = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            pred = output.max(1, keepdim=True)[1]
            train_acc += pred.eq(batch_y.view_as(pred)).sum().item()
            # batch_y=torch.nn.functional.one_hot(batch_y)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()




