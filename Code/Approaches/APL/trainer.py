import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def train(self, model, optimizer, criterion):
        model.train()
        # prediction = []
        # true = []
        for data, labels in self.data_loader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            pred, y=self.train_batch(data, labels, model, criterion, optimizer)
        #     pred=F.softmax(pred, dim=1)
        #     _, pred = torch.max(pred.data, 1)
        #     for i in range(len(pred)):
        #         prediction.append(pred.cpu()[i])
        #         true.append(y.cpu()[i])
        # return prediction, true

    def train_batch(self, x, y, model, criterion, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        return pred, y