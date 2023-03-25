# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
import tools
from models import Matrix_optimize
import copy
def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm

def train_m(V, r, k, e):
    m, n = np.shape(V)
    W = np.mat(np.random.random((m, r)))
    H = np.mat(np.random.random((r, n)))
    data = []

    for x in range(k):
        V_pre = np.dot(W, H)
        E = V - V_pre
        err = 0.0
        err = np.sum(np.square(E))
        data.append(err)
        if err < e:  # threshold
            break

        a = np.dot(W.T, V)  # Hkj
        b = np.dot(np.dot(W.T, W), H)

        for i_1 in range(r):
            for j_1 in range(n):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = np.dot(V, H.T)
        d = np.dot(np.dot(W, H), H.T)
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]

        W = norm(W)

    return W, H, data


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

def train(model, train_loader, epoch, optimizer, criterion):
    train_total = 0
    train_correct = 0

    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, logits = model(data, revision=False)
        prec1, = accuracy(logits, labels, topk=(1,))
        train_total += 1
        train_correct += prec1
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)

    return train_acc


def train_correction(model, train_loader, epoch, optimizer, W_group, basis_matrix_group, batch_size, num_classes,
                     basis):
    train_total = 0
    train_correct = 0

    for i, (data, labels) in enumerate(train_loader):
        loss = 0.
        data = data.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, logits = model(data, revision=False)

        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = torch.from_numpy(matrix).float().cuda()
            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = Variable(pro1 / pro2, requires_grad=True)
            logits_correction = torch.log(logits_correction + 1e-12)
            logits_single = torch.log(logits_single + 1e-12)
            loss_ = beta * F.nll_loss(logits_single, labels[j].unsqueeze(0))
            loss += loss_
            logits_correction_total[j, :] = logits_correction
        logits_correction_total = logits_correction_total.cuda()
        loss = loss / len(labels)
        prec1, = accuracy(logits_correction_total, labels, topk=(1,))
        train_total += 1
        train_correct += prec1
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc


def val_correction(model, val_loader, epoch, W_group, basis_matrix_group, batch_size, num_classes, basis):
    val_total = 0
    val_correct = 0

    loss_total = 0.
    for i, (data, labels) in enumerate(val_loader):

        data = data.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        loss = 0.
        _, logits = model(data, revision=False)

        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = norm(matrix)
            matrix = torch.from_numpy(matrix).float().cuda()

            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = Variable(pro1 / pro2, requires_grad=False)
            logits_correction = torch.log(logits_correction + 1e-8)
            loss_ = beta * F.nll_loss(logits_correction, labels[j].unsqueeze(0))
            if torch.isnan(loss_) == True:
                loss_ = 0.
            loss += loss_
            logits_correction_total[j, :] = logits_correction

        logits_correction_total = logits_correction_total.cuda()
        loss = loss / len(labels)
        prec1, = accuracy(logits_correction_total, labels, topk=(1,))
        val_total += 1
        val_correct += prec1

        try:
            loss_total += loss.item()
        except:
            loss_total += loss

    val_acc = float(val_correct) / float(val_total)

    return val_acc


def train_revision(model, train_loader, epoch, optimizer, W_group, basis_matrix_group, batch_size, num_classes, basis):
    train_total = 0
    train_correct = 0
    true = []
    prediction = []
    for i, (data, labels) in enumerate(train_loader):

        data = data.cuda()
        labels = labels.cuda()
        loss = 0.
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, logits, revision = model(data, revision=True)

        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = torch.from_numpy(matrix).float().cuda()
            matrix = tools.norm(matrix + revision)

            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = pro1 / pro2
            logits_correction = torch.log(logits_correction + 1e-12)
            logits_single = torch.log(logits_single + 1e-12)
            loss_ = beta * F.nll_loss(logits_single, labels[j].unsqueeze(0))
            loss += loss_
            logits_correction_total[j, :] = logits_correction
        logits_correction_total = logits_correction_total.cuda()
        loss = loss / len(labels)
        prec1, = accuracy(logits_correction_total, labels, topk=(1,))
        outputs = F.softmax(logits_correction_total, dim=1)
        _, pred = torch.max(outputs.data, 1)
        labels=labels.cpu()
        pred=pred.cpu()
        for i in range(len(pred)):
            true.append(labels[i])
            prediction.append(pred[i])
        train_total += 1
        train_correct += prec1

        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc,prediction,true


def val_revision(model, train_loader, epoch, W_group, basis_matrix_group, batch_size, num_classes, basis):
    val_total = 0
    val_correct = 0

    for i, (data, labels) in enumerate(train_loader):
        model.eval()
        data = data.cuda()
        labels = labels.cuda()
        loss = 0.
        # Forward + Backward + Optimize

        _, logits, revision = model(data, revision=True)

        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = torch.from_numpy(matrix).float().cuda()
            matrix = tools.norm(matrix + revision)
            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = Variable(pro1 / pro2, requires_grad=True)
            logits_correction = torch.log(logits_correction + 1e-12)
            loss_ = beta * F.nll_loss(logits_correction, labels[j].unsqueeze(0))
            loss += loss_
            logits_correction_total[j, :] = logits_correction
        logits_correction_total = logits_correction_total.cuda()
        prec1, = accuracy(logits_correction_total, labels, topk=(1,))
        val_total += 1
        val_correct += prec1

    val_acc = float(val_correct) / float(val_total)

    return val_acc


# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels in test_loader:
        data = data.cuda()
        _, logits = model(data, revision=False)
        outputs = F.softmax(logits, dim=1)
        _, pred1 = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()
    acc = float(correct1) / float(total1)


def respresentations_extract(train_loader, model, num_sample, dim_respresentations, batch_size):
    model.eval()
    A = torch.rand(num_sample, dim_respresentations)
    ind = int(num_sample / batch_size)
    with torch.no_grad():
        for i, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            logits, _ = model(data, revision=False)
            if i < ind:
                A[i * batch_size:(i + 1) * batch_size, :] = logits
            else:
                A[ind * batch_size:, :] = logits

    return A.cpu().numpy()


def probability_extract(train_loader, model, num_sample, num_classes, batch_size):
    model.eval()
    A = torch.rand(num_sample, num_classes)
    ind = int(num_sample / batch_size)
    with torch.no_grad():
        for i, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            _, logits = model(data, revision=False)
            logits = F.softmax(logits, dim=1)
            if i < ind:
                A[i * batch_size:(i + 1) * batch_size, :] = logits
            else:
                A[ind * batch_size:, :] = logits

    return A.cpu().numpy()


def estimate_matrix(logits_matrix, model_save_dir, basis=10, num_classes=2):
    transition_matrix_group = np.empty((basis, num_classes, num_classes))
    idx_matrix_group = np.empty((num_classes, basis))
    a = np.linspace(97, 99, basis)
    a = list(a)
    for i in range(len(a)):
        percentage = a[i]
        index = int(i)
        logits_matrix_ = copy.deepcopy(logits_matrix)
        transition_matrix, idx = tools.fit(logits_matrix_, num_classes, percentage, True)
        transition_matrix = norm(transition_matrix)
        idx_matrix_group[:, index] = np.array(idx)
        transition_matrix_group[index] = transition_matrix
    idx_group_save_dir = model_save_dir + '/' + 'idx_group.npy'
    group_save_dir = model_save_dir + '/' + 'T_group.npy'
    np.save(idx_group_save_dir, idx_matrix_group)
    np.save(group_save_dir, transition_matrix_group)
    return idx_matrix_group, transition_matrix_group


def basis_matrix_optimize(model, optimizer, basis, num_classes, W_group, transition_matrix_group, idx_matrix_group,
                          func, model_save_dir, epochs):
    basis_matrix_group = np.empty((basis, num_classes, num_classes))

    for i in range(num_classes):

        model = tools.init_params(model)
        for epoch in range(epochs):
            loss_total = 0.
            for j in range(basis):
                class_1_idx = int(idx_matrix_group[i, j])
                W = list(np.array(W_group[class_1_idx, :]))
                T = torch.from_numpy(transition_matrix_group[j, i, :][:, np.newaxis]).float()
                prediction = model(W[0], num_classes)
                optimizer.zero_grad()
                loss = func(prediction, T)
                loss.backward()
                optimizer.step()
                loss_total += loss
            if loss_total < 0.02:
                break

        for x in range(basis):
            parameters = np.array(model.basis_matrix[x].weight.data)

            basis_matrix_group[x, i, :] = parameters
    A_save_dir = model_save_dir + '/' + 'A.npy'
    np.save(A_save_dir, basis_matrix_group)
    return basis_matrix_group


def matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis):
    coefficient = W_group[idx, :]

    M = np.zeros((num_classes, num_classes))
    for i in range(basis):
        temp = float(coefficient[0, i]) * basis_matrix_group[i, :, :]
        M += temp
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] < 1e-6:
                M[i, j] = 0.
    return M


def PDN(x_train,y_train,x_test,y_test,y_train_noise,model,n_epoch_1, n_epoch_2, n_epoch_3, n_epoch_4, dim, batch_size,
        basis, iteration_nmf, num_classes):
    model_save_dir = "models"

    train_dataset = Data.TensorDataset(torch.tensor(x_train[:int(0.9 * len(x_train))]),
                                       torch.tensor(y_train_noise[:int(0.9 * len(y_train_noise))]))
    val_dataset = Data.TensorDataset(torch.tensor(x_train[int(0.9 * (len(x_train))):]),
                                     torch.tensor(y_train[int(0.9 * len(y_train)):]))
    test_dataset = Data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               drop_last=False,
                                               shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             drop_last=False,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              drop_last=False,
                                              shuffle=False)
    # Define models
    # print('building model...')

    clf1 = model

    clf1.cuda()
    optimizer = torch.optim.SGD(clf1.parameters(), lr=0.01,momentum=0.9)

    best_acc = 0.0
    # training
    for epoch in range(1, n_epoch_1):
        # train models
        clf1.train()
        train_acc = train(clf1, train_loader, epoch, optimizer, nn.CrossEntropyLoss())
        # validation
        val_acc, _, _, _, _,_= evaluate(val_loader, clf1)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(clf1.state_dict(), 'models/model.pth')

    # print('Matrix Factorization is doing...')
    clf1.load_state_dict(torch.load('models/model.pth'))
    A = respresentations_extract(train_loader, clf1, len(train_dataset), dim, batch_size)
    A_val = respresentations_extract(val_loader, clf1, len(val_dataset), dim, batch_size)
    A_total = np.append(A, A_val, axis=0)
    W_total, H_total, error = train_m(A_total, basis, iteration_nmf, 1e-5)
    for i in range(W_total.shape[0]):
        for j in range(W_total.shape[1]):
            if W_total[i, j] < 1e-6:
                W_total[i, j] = 0.
    W = W_total[0:len(train_dataset), :]
    W_val = W_total[len(train_dataset):, :]
    # print('Transition Matrix is estimating...Wating...')
    logits_matrix = probability_extract(train_loader, clf1, len(train_dataset), num_classes, batch_size)
    idx_matrix_group, transition_matrix_group = estimate_matrix(logits_matrix, model_save_dir)
    logits_matrix_val = probability_extract(val_loader, clf1, len(val_dataset), num_classes, batch_size)
    idx_matrix_group_val, transition_matrix_group_val = estimate_matrix(logits_matrix_val, model_save_dir)
    func = nn.MSELoss()

    model = Matrix_optimize(basis, num_classes)
    optimizer_1 = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    basis_matrix_group = basis_matrix_optimize(model, optimizer_1, basis, num_classes, W,
                                               transition_matrix_group, idx_matrix_group, func, model_save_dir,
                                               n_epoch_4)

    basis_matrix_group_val = basis_matrix_optimize(model, optimizer_1, basis, num_classes, W_val,
                                                   transition_matrix_group_val, idx_matrix_group_val, func,
                                                   model_save_dir, n_epoch_4)

    for i in range(basis_matrix_group.shape[0]):
        for j in range(basis_matrix_group.shape[1]):
            for k in range(basis_matrix_group.shape[2]):
                if basis_matrix_group[i, j, k] < 1e-6:
                    basis_matrix_group[i, j, k] = 0.

    optimizer_ = torch.optim.SGD(clf1.parameters(), lr=0.01,momentum=0.9)

    best_acc = 0.0
    for epoch in range(1, n_epoch_2):
        # train model
        clf1.train()

        train_acc = train_correction(clf1, train_loader, epoch, optimizer_, W, basis_matrix_group, batch_size,
                                     num_classes, basis)
        # validation
        val_acc = val_correction(clf1, val_loader, epoch, W_val, basis_matrix_group_val, batch_size, num_classes, basis)

        # evaluate models
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(clf1.state_dict(), 'models/model.pth')

    clf1.load_state_dict(torch.load('models/model.pth'))
    optimizer_r = torch.optim.SGD(clf1.parameters(), lr=0.01, weight_decay=1e-4,momentum=0.9)
    nn.init.constant_(clf1.T_revision.weight, 0.0)
    for epoch in range(1, n_epoch_3):
        # train models

        clf1.train()
        train_revision(clf1, train_loader, epoch, optimizer_r, W, basis_matrix_group, batch_size,
                                   num_classes, basis)


        # evaluate models

    evaluate(test_loader, clf1)




