import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()#(3,224,224)
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=0),#(32,220,220)
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2) #(32,110,110)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=7,stride=1,padding=0),#(64,104,104)
            nn.ReLU(),
            nn.MaxPool2d(2)#output shape (64,52,52)
        )
        self.out = nn.Sequential(
            nn.Linear(64*52*52,256),
            nn.Linear(256,7),
            #nn.Sigmoid()
        )
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.out(out)
        return out

if(__name__=="__main__"):
    print("Model structure:\n")
    net = Net()
    print(net)