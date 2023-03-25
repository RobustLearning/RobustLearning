import torch.nn.functional as F
from torch import optim
import numpy as np
import torch.utils.data as Data
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta):
    '''
    The LRT correction scheme.
    pred_softlabels_bar is the prediction of the network which is compared with noisy label y_tilde.
    If the LR is smaller than the given threshhold delta, we reject LRT and flip y_tilde to prediction of pred_softlabels_bar

    Input
    pred_softlabels_bar: rolling average of output after softlayers for past 10 epochs. Could use other rolling windows.
    y_tilde: noisy labels at current epoch
    delta: LRT threshholding
    Output
    y_tilde : new noisy labels after cleanning
    clean_softlabels : softversion of y_tilde
    '''
    ntrain = pred_softlabels_bar.shape[0]
    num_class = pred_softlabels_bar.shape[1]
    for i in range(ntrain):
        cond_1 = (not pred_softlabels_bar[i].argmax()==y_tilde[i])
        cond_2 = (pred_softlabels_bar[i].max()/pred_softlabels_bar[i][y_tilde[i]] > delta)
        if cond_1 and cond_2:
            y_tilde[i] = pred_softlabels_bar[i].argmax()
    eps = 1e-2
    clean_softlabels = torch.ones(ntrain, num_class)*eps/(num_class - 1)
    clean_softlabels.scatter_(1, torch.tensor(np.array(y_tilde)).reshape(-1, 1), 1 - eps)
    return y_tilde, clean_softlabels

def updateA(s, h, rho=0.9):
    '''
    Used to calculate retroactive loss

    Input
    s : output after softlayer of NN for a specific former epoch
    h : logrithm of output after softlayer of NN at current epoch

    Output
    result : retroactive loss L_retro
    A : correction matrix
    '''
    eps = 1e-4
    h = torch.tensor(h, dtype=torch.float32).reshape(-1, 1)
    s = torch.tensor(s, dtype=torch.float32).reshape(-1, 1)
    A = torch.ones(len(s), len(s))*eps
    A[s.argmax(0)] = rho - eps/(len(s)-1)
    result = -((A.matmul(s)).t()).matmul(h)

    return result, A


def LRT(data_root,train_dataset,test_dataset,model,epochs,lr,batch_size,num_class,every_n_epoch,epoch_update,epoch_start,epoch_interval):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    trainloader=Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    testloader=Data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    eps = 1e-6
    n_train=len(train_dataset)

    noise_y_train=np.load(data_root+ "train_label_noisy.npy")
    noise_softlabel=torch.ones(n_train,num_class)*eps/(num_class-1)
    noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1 - eps)
    train_dataset.update_corrupted_softlabel(noise_softlabel)

    net_trust= model.to(device)

    A=1/num_class*torch.ones(n_train,num_class,num_class,requires_grad=False).float().to(device)
    h=np.zeros([n_train,num_class])


    criterion_1=nn.NLLLoss()
    pred_softlabels=np.zeros([n_train,every_n_epoch,num_class],dtype=np.float32)


    for epoch in range(epochs):
        train_correct=0
        train_loss=0
        train_total=0
        delta=1.2+0.02*max(epoch-epoch_update+1,0)


        optimizer_trust = optim.SGD(net_trust.parameters(), lr=lr, momentum=0.9)

        net_trust.train()

        #train with noisy data
        for sentences,labels,softlabels,indices in trainloader:
            sentences,labels,softlabels=sentences.to(device),labels.to(device),softlabels.to(device)
            outputs=net_trust(sentences)
            log_outputs=torch.log_softmax(outputs,1).float()
            logits = F.softmax(outputs, dim=1)
            _, pred = torch.max(logits.data, 1)
            if epoch in [epoch_start-1,epoch_start+epoch_interval-1]:
                h[indices]=log_outputs.detach().cpu()
            normal_outputs=torch.softmax(outputs,1)

            if epoch>=epoch_start:
                A_batch = A[indices].to(device)
                loss = sum([-A_batch[i].matmul(softlabels[i].reshape(-1, 1).float()).t().matmul(log_outputs[i])
                            for i in range(len(indices))]) / len(indices) + criterion_1(log_outputs, labels)
            else:  # use loss_ce
                loss = criterion_1(log_outputs, labels)

            optimizer_trust.zero_grad()
            loss.backward()
            optimizer_trust.step()

            if epoch >= (epoch_update-every_n_epoch):
                pred_softlabels[indices, epoch % every_n_epoch, :] = normal_outputs.detach().cpu().numpy()

            train_loss+=loss.item()
            train_total+=len(sentences)
            _,predicted=outputs.max(1)

            if epoch in [epoch_start-1,epoch_start+epoch_interval-1]:
                unsolved=0
                infeasible=0
                y_soft=train_dataset.get_data_softlabel()

                with torch.no_grad():
                    for i in range(n_train):
                        try:
                            result, A_opt = updateA(y_soft[i], h[i], rho=0.9)
                        except:
                            A[i] = A[i]
                            unsolved += 1
                            continue

                        if (result == np.inf):
                            A[i] = A[i]
                            infeasible += 1
                        else:
                            A[i] = torch.tensor(A_opt)

            if epoch >= epoch_update:
                y_tilde=train_dataset.get_data_labels()
                pred_softlabels_bar=pred_softlabels.mean(1)
                clean_labels, clean_softlabels = lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta)
                train_dataset.update_corrupted_softlabel(clean_softlabels)
                train_dataset.update_corrupted_label(clean_softlabels.argmax(1))

    test_correct=0
    net_trust.eval()
    with torch.no_grad():
        for sentences,labels,softlabels,indices in testloader:
            sentences,labels=sentences.to(device),labels.to(device)
            outputs=net_trust(sentences)

            test_correct+=outputs.argmax(1).eq(labels).sum().item()
        test_acc=test_correct/len(testloader.dataset)


