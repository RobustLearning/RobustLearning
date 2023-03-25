
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Evaluator():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def eval(self, model):
        model.eval()
        # preds = []
        # targets = []
        # test_correct = 0
        # prediction = []
        # true = []
        for i, (data, labels) in enumerate(self.data_loader):
            pred = self.eval_batch(x=data, y=labels, model=model)
        #     test_correct += pred.cpu().eq(labels.cpu().view_as(pred.cpu())).sum().item()
        #     preds.append(pred.cpu().clone().numpy().tolist())
        #     targets.append(labels.cpu().clone().numpy().tolist())
        #
        # test_acc = test_correct / len(self.data_loader.dataset)
        # for i in range(len(preds)):
        #     for j in range(len(preds[i])):
        #         prediction.append(preds[i][j])
        #         true.append(targets[i][j])
        # prediction = np.array(prediction)
        # true = np.array(true)
        # f1 = f1_score(true, prediction, zero_division=0)
        # precision = precision_score(true, prediction, zero_division=0)
        # recall = recall_score(true, prediction, zero_division=0)

        # return test_acc, f1, precision, recall, prediction,true

    def eval_batch(self, x, y, model):
        model.eval()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = model(x)
        _, pred = torch.max(outputs.data, 1)
        return pred
