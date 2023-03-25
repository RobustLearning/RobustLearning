from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def clean(X,Y,threshold=0.5):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    rf = RandomForestClassifier(n_estimators=50, n_jobs=4)

    probabilities = np.zeros(Y.shape[0], dtype=float)
    for train_index, test_index in skf.split(X, Y):
        rf.fit(X[train_index], Y[train_index])
        probs = rf.predict_proba(X[test_index])
        probabilities[test_index] = probs[range(len(test_index)), Y[test_index]]

    hardness = 1 - probabilities
    clean_idx = hardness <= threshold
    Xt, Yt = X[clean_idx], Y[clean_idx]
    return Xt, Yt

def RFF(train_data,train_label,test_data,test_label,model,batch_size,epochs,learning_rate):
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





