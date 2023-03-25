import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1=torch.nn.Linear(65,45)
        self.fc2=torch.nn.Linear(45,30)
        self.fc3=torch.nn.Linear(30,20)
        self.fc4=torch.nn.Linear(20,2)
        self.drop=torch.nn.Dropout(0.5)

    def forward(self,data):
        data=data.view(-1,65)
        out1=torch.nn.functional.relu(self.drop(self.fc1(data)))
        out2=torch.nn.functional.relu(self.drop(self.fc2(out1)))
        out3=torch.nn.functional.relu(self.drop(self.fc3(out2)))
        out4=torch.nn.functional.softmax(self.fc4(out3))
        return out4.view(-1,2)