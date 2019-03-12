import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)
    
class CrossBlock(nn.Module):
    def __init__(self,inChannels,outChannels):
        super(CrossBlock,self).__init__()
        self.conv1 = conv3x3(inChannels,outChannels)
        self.bn = nn.BatchNorm2d(outChannels)
        self.conv2 = conv1x1(inChannels,outChannels)
    def forward(self,x):
        out1 = self.conv1(x)
        out1 = self.bn(out1)
        out2 = self.conv2(x)
        out = out1 + out2
        return out
        
class CCTnet(nn.Module):
    def __init__(self, num_classes=12):
        super(CCTnet, self).__init__()
        self.features = nn.Sequential(
            CrossBlock(3,64),
            nn.ReLU(True),
            CrossBlock(64,64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CrossBlock(64,128),
            nn.ReLU(True),
            CrossBlock(128,128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CrossBlock(128,256),
            nn.ReLU(True),
            CrossBlock(256,256),
            nn.ReLU(True),
            CrossBlock(256,256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CrossBlock(256,512),
            nn.ReLU(True),
            CrossBlock(512,512),
            nn.ReLU(True),
            CrossBlock(512,512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CrossBlock(512,512),
            nn.ReLU(True),
            CrossBlock(512,512),
            nn.ReLU(True),
            CrossBlock(512,512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.lstm = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.LSTM(1024, 512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1, -1)
        x = self.lstm(x)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
def cctnet(**kwargs):
    model = CCTnet(**kwargs)
    return model
    
if __name__ == "__main__":
    net = cctnet(num_classes = 12)
    image = torch.rand([1,3,224,224])
    net.train()
    output = net(image)
    print(output)