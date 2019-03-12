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
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.conv1 = conv1x1(inChannels,outChannels)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = conv3x3(outChannels,outChannels)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.conv3 = conv1x1(outChannels,outChannels)
        self.bn3 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        if(self.inChannels == self.outChannels):
            cross = x
        else:
            cross = self.conv1(x)
            cross = self.bn1(cross)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += cross
        out = self.relu(out)
        return out
        
class CCnet(nn.Module):
    def __init__(self, num_classes=12):
        super(CCnet, self).__init__()
        self.features = nn.Sequential(
            #3 * 224 * 224
            CrossBlock(3,64),
            CrossBlock(64,64),
            nn.MaxPool2d(kernel_size=2, stride=2),#64 * 112 * 112
            CrossBlock(64,128),
            CrossBlock(128,128),
            nn.MaxPool2d(kernel_size=2, stride=2),#128 * 56 * 56
            CrossBlock(128,256),
            CrossBlock(256,256),
            CrossBlock(256,256),
            nn.MaxPool2d(kernel_size=2, stride=2),#256 * 28 * 28
            CrossBlock(256,512),
            CrossBlock(512,512),
            CrossBlock(512,512),
            nn.MaxPool2d(kernel_size=2, stride=2),#512 * 14 * 14
            CrossBlock(512,512),
            CrossBlock(512,512),
            CrossBlock(512,512),
            nn.MaxPool2d(kernel_size=2, stride=2),#512 * 7 * 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, num_classes),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
def ccnet(**kwargs):
    model = CCnet(**kwargs)
    return model
    
if __name__ == "__main__":
    net = ccnet(num_classes = 12)
    image = torch.rand([1,3,224,224])
    net.train()
    output = net(image)
    print(output)