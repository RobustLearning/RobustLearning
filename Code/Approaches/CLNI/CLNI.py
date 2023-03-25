from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import _get_weights
import numpy as np
import torch
import torch.nn.functional as F
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


def KDN(X,Y,K=5,n_jobs=-1,weight='uniform'):
    knn=KNeighborsClassifier(n_neighbors=K,n_jobs=n_jobs,weights=weight).fit(X,Y)
    dist,kid = knn.kneighbors()
    weights = _get_weights(dist,weight)
    if weights is None:
        weights = np.ones_like(kid)
    disagreement = Y[kid] != Y.reshape(-1, 1)
    return np.average(disagreement, axis=1, weights=weights)


def clean(X,Y):
    N,alpha=5,0.6
    Xt,Yt=X.copy(),Y.copy()
    while True:
        ne=KDN(X,Y)
        cidx=ne<=alpha
        N=len(Xt)
        Xt,Yt=Xt[cidx],Y[cidx]
        if cidx.sum()/N>=0.99:
            break
        return Xt,Yt

def CLNI(train_data,train_label,test_data,test_label,model,batch_size,epochs,learning_rate):
    train_data,train_label=clean(train_data,train_label)
    train_dataset = Data.TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
    test_dataset = Data.TensorDataset(torch.tensor(test_data), torch.tensor(test_label))

    kwargs = {
        'num_workers': 4,
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


