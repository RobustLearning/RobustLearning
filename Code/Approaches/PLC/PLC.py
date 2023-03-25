import torch
import torch.nn.parallel
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
import torch.nn.functional as F
from utils import lrt_correction


# @profile
def PLC(trainset,testset,model,num_epoch,batch_size,num_workers,num_class,rollWindow,lr,current_delta,warm_up,inc):
    torch.backends.cudnn.deterministic = True  # need to set to True as well
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # -- set network, optimizer, scheduler, etc
    net = model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    net = net.to(device)

    f_record = torch.zeros([rollWindow, len(trainset), num_class])
    for epoch in range(num_epoch):
        net.train()
        for features, labels, _, indices in trainloader:
            if features.shape[0] == 1:
                continue
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            logits = F.softmax(outputs, dim=1)
            _, pred = torch.max(logits.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            f_record[epoch % rollWindow, indices] = F.softmax(outputs.detach().cpu(), dim=1)

        if epoch >= warm_up:
            f_x = f_record.mean(0)
            y_tilde = trainset.targets
            y_corrected, current_delta = lrt_correction(np.array(y_tilde).copy(), f_x, current_delta=current_delta, delta_increment=inc)

            trainset.update_corrupted_label(y_corrected)

        scheduler.step()

    # -- Final testing
    test_total = 0
    test_correct = 0
    net.eval()
    with torch.no_grad():
        for i, (sentences, labels, _, _) in enumerate(testloader):
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = net(sentences)
            test_total += sentences.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()




