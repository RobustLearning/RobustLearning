import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from losses import get_loss
from utils import get_scheduler, get_optimizer
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SAT(train_loader, test_loader, num_classes, targets, model, epochs=50):
    model = model.to(device)
    cudnn.benchmark = True
    loss = "sat"
    optim = "sgd"
    sche = "step"
    criterion = get_loss(loss, labels=targets, num_classes=num_classes)
    optimizer = get_optimizer(model, optim)
    scheduler = get_scheduler(optimizer, sche)
    for epoch in range(epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step(epoch)
    validate(test_loader, model)





def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """

    # switch to train mode
    model.train()
    # end = time.time()
    for input, target, index in train_loader:

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target, index, epoch)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs = F.softmax(output, dim=1)
        _, pred = torch.max(outputs.data, 1)


def validate(val_loader, model):
    """
    Run evaluation
    """
    model.eval()
    correct = 0
    # end = time.time()
    for input, target in val_loader:
        input = input.to(device)
        target = target.to(device)
        # compute output
        with torch.no_grad():
            output = model(input)
            # loss = F.cross_entropy(output, target)
            _, pred = torch.max(output.data, 1)
            correct += pred.eq(target).sum().item()

    test_score = correct / len(val_loader.dataset)

