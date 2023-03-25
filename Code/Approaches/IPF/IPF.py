import numpy as np
import torch.nn.functional as F
from sklearn.utils import shuffle
import torch
from sklearn.tree import DecisionTreeClassifier
import torch.utils.data as Data

def evaluate(test_loader,model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_score=correct/len(test_loader.dataset)

def clean(X,Y,n=5,max_iter=3):
    Xt, Yt = shuffle(X, Y)
    orig_size = len(X)
    n_iters_with_small_change = 0
    tmp = 0
    while n_iters_with_small_change < max_iter:
        tmp += 1
        cur_size = len(Xt)
        breaks = [(len(Xt) // n) * i for i in range(1, n)]
        Xs, Ys = np.split(Xt, breaks), np.split(Yt, breaks)

        clfs = []
        for i in range(n):
            c = DecisionTreeClassifier(max_depth=2).fit(Xs[i], Ys[i])
            clfs.append(c)

        preds = np.zeros((len(Xt), n))
        for i in range(n):
            preds[:, i] = clfs[i].predict(Xt)
        eqs = preds == Yt.reshape(-1, 1)  # Shape: (len(Xt),self.n)
        clean_idx = eqs.sum(axis=1) >= (n / 2)  # Idx of clean samples

        Xt, Yt = Xt[clean_idx], Yt[clean_idx]

        cur_change = cur_size - len(Xt)
        if cur_change <= .01 * orig_size:
            n_iters_with_small_change += 1
        else:
            n_iters_with_small_change = 0  # Because these small change has to be consecutively 3 times
        # print(tmp,cur_change,orig_size,cur_change/orig_size)
    return Xt, Yt

def IPF(train_data,train_label,test_data,test_label,model,batch_size,epochs,learning_rate):
    train_data,train_label=clean(train_data,train_label)
    train_dataset = Data.TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
    test_dataset = Data.TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
    kwargs = {
        'num_workers': 8,
        'pin_memory': True
    } if torch.cuda.is_available() else {}
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    model=model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model.train()
    for epoch in range(epochs):
        for data, labels in train_loader:
            data = data.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(data)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            loss = criterion(logits, labels.long())
            loss.backward()
            optimizer.step()
    evaluate(test_loader, model)
