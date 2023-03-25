import torch.optim as optim
import torch.utils.data as Data
from utils import get_softmax_out
from ops import *


def SEAL(train_dataset,test_dataset,train_dataset_soft,model,seal,epochs,save,batch_size,lr):
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    softmax_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    # results
    # results_root = "../Data/results/SEAL/"+file+"/"+exp_time+"/SEAL="+str(seal)
    # if not os.path.isdir(results_root):
    #     os.makedirs(results_root)
    """ Get softmax_out_avg - normal training on noisy labels """
    if seal == 0:
        # Building model
        model = model.to(device)
        # Training
        softmax_out_avg = np.zeros([len(train_dataset), num_classes])
        for epoch in range(1, epochs + 1):
            optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
            # optimizer=optim.Adam(model.parameters(), lr=1e-3)
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            softmax_out_avg += get_softmax_out(model, softmax_loader, device)

        softmax_out_avg /= epochs
        if save:
            softmax_root = "../Data/results/SEAL"+'/SEAL=0'+'/softmax_out_avg.npy'
            np.save(softmax_root, softmax_out_avg)
            print('new softmax_out_avg saved to', softmax_root, ', shape: ', softmax_out_avg.shape)

    """ Self Evolution - training on softmax_out_avg """
    if seal >= 1:
        # Loading softmax_out_avg of last phase
        softmax_root ="../Data/results/SEAL"+"/SEAL="+str(seal-1)+'/softmax_out_avg.npy'
        softmax_out_avg = np.load(softmax_root)
        print('softmax_out_avg loaded from', softmax_root, ', shape: ', softmax_out_avg.shape)

        # Dataset with soft targets
        train_loader_soft = torch.utils.data.DataLoader(train_dataset_soft, batch_size=batch_size, shuffle=True,
                                                        **kwargs)
        # Building model
        model = model.to(device)
        # Training
        softmax_out_avg = np.zeros([len(train_dataset), num_classes])
        for epoch in range(1, epochs + 1):
            optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
            # optimizer=optim.Adam(model.parameters(), lr=1e-3)
            train_soft(model, device, train_loader_soft, optimizer, epoch)
            test(model, device, test_loader)
            softmax_out_avg += get_softmax_out(model, softmax_loader, device)


        softmax_out_avg /= epochs
        if save:
            softmax_root ="../Data/results/SEAL"+'/SEAL='+str(seal)+'/softmax_out_avg.npy'
            np.save(softmax_root, softmax_out_avg)
            print('new softmax_out_avg saved to', softmax_root, ', shape: ', softmax_out_avg.shape)


