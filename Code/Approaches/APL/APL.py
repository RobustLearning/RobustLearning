import torch.utils.data as Data
from trainer import *
from evaluator import *
from loss import *
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(epochs,model,optimizer, scheduler, criterion, trainer):
    for epoch in range(epochs):
        trainer.train(model, optimizer, criterion)
        scheduler.step()



def APL(x_train,y_train,x_test,y_test,model,loss_type,epochs,num_class,batch_size,learning_rate,alpha,beta):

    train_dataset = Data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = Data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

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

    net = model.to(device)

    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0)

    if loss_type==1:
        criterion=NFLandMAE(alpha=alpha,beta=beta,num_classes=num_class,gamma=0.5)

    elif loss_type==2:
        criterion=NFLandRCE(alpha=alpha,beta=beta,num_classes=num_class,gamma=0.5)

    elif loss_type==3:
        criterion=NCEandMAE(alpha=alpha,beta=beta,num_classes=num_class)

    elif loss_type==4:
        criterion=NCEandRCE(alpha=alpha,beta=beta,num_classes=num_class)

    else:
        criterion=NFLandMAE(alpha=alpha,beta=beta,num_classes=num_class,gamma=0.5)


    trainer = Trainer(train_loader)
    evaluator = Evaluator(test_loader)
    train(epochs,net,optimizer,scheduler,criterion,trainer)
    evaluator.eval(net)

