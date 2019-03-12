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
    
class Inception(nn.Module):
    def __init__(self,inChannels,outChannels):
        super(Inception,self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.conv1 = conv1x1(inChannels,outChannels)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = conv3x3(outChannels,outChannels)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.conv3 = conv3x3(outChannels,outChannels)
        self.bn3 = nn.BatchNorm2d(outChannels)
        self.conv4 = conv1x1(inChannels,outChannels)
        self.bn4 = nn.BatchNorm2d(outChannels)
        self.conv5 = conv3x3(outChannels,outChannels)
        self.bn5 = nn.BatchNorm2d(outChannels)
        self.conv6 = conv1x1(inChannels,outChannels)
        self.bn6 = nn.BatchNorm2d(outChannels)
        #self.conv7 = conv1x1(inChannels,outChannels)
        #self.bn7 = nn.BatchNorm2d(outChannels)
        #self.pool = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn3(out1)
        out1 = self.relu(out1)
        
        out2 = self.conv4(x)
        out2 = self.bn4(out2)
        out2 = self.relu(out2)
        out2 = self.conv5(out2)
        out2 = self.bn5(out2)
        out2 = self.relu(out2)
        
        #out3 = self.pool(x)
        out3 = self.conv6(x)
        out3 = self.bn6(out3)
        out3 = self.relu(out3)
        
        #out4 = self.conv7(x)
        #out4 = self.bn7(out4)
        #out4 = self.relu(out4)
        
        out = out1 + out2 + out3# + out4
        #outputs = [out1,out2,out3,out4]
        #out = torch.cat(outputs,1)
        return out
        
class Incep(nn.Module):
    def __init__(self, num_classes=12):
        super(Incep, self).__init__()
        self.features = nn.Sequential(
            #3 * 224 * 224
            Inception(3,64),
            Inception(64,64),
            nn.MaxPool2d(kernel_size=2, stride=2),#64 * 112 * 112
            Inception(64,128),
            Inception(128,128),
            nn.MaxPool2d(kernel_size=2, stride=2),#128 * 56 * 56
            Inception(128,256),
            Inception(256,256),
            Inception(256,256),
            nn.MaxPool2d(kernel_size=2, stride=2),#256 * 28 * 28
            Inception(256,512),
            Inception(512,512),
            Inception(512,512),
            nn.MaxPool2d(kernel_size=2, stride=2),#512 * 14 * 14
            Inception(512,512),
            Inception(512,512),
            Inception(512,512),
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
        
def incep(**kwargs):
    model = Incep(**kwargs)
    return model
    
if __name__ == "__main__":
    net = incep(num_classes = 12)
    image = torch.rand([1,3,224,224])
    net.train()
    output = net(image)
    print(output)