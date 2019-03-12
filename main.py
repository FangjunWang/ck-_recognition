import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from data_loader import DataSet
import config as cfg

from model.alexnet import alexnet
from model.vggnet import vgg16
from model.vgg_f import vgg_f
from model.ccnet import ccnet
from model.cctnet import cctnet
from model.incep import incep
from model.net import Net
from model.region_alex import region_alex

#loss function
class myloss(nn.Module):
    def __init__(self):
        super(myloss, self).__init__()
        return
    
    def forward(self, input, target):
        size = input.size(0)
        #loss = ((input.float()-target.float())*(input.float()-target.float())).sum()/size
        pos_to_log = input[target == 1]
        pos_to_log[pos_to_log.data == 0] = 1e-20
        pos_part = torch.log(pos_to_log).sum()
        neg_to_log = 1 - input[target < 1]
        neg_to_log[neg_to_log.data == 0] = 1e-20
        neg_part = torch.log(neg_to_log).sum()
        loss = abs(pos_part + neg_part)/size
        return loss
        
def train(train_loader, model,criterion, optimizer, epoch, logfile):
    #train definition
    model.train()
    
    for i, (input, target) in enumerate(train_loader):
        #target = torch.reshape(target,[target.size(0)])
        #print(target)
        input_var = Variable(input.cuda(async=True))
        target_var = Variable(target.cuda(async=True))
        optimizer.zero_grad()
        
        output = model(input_var)
        #print(output)
        
        loss = criterion(output, target_var)
        #statistics_list = statistics(output[:][:10],target[][:10])
        #mean_f1_score, f1_score_list = f1_score(statistics_list)
        loss.backward()
        optimizer.step()
        with open(logfile,'a')as fp:
            fp.write('\nEpoch: {} Loss: {}'.format(epoch+1,loss))
        print('Epoch: {}-{}/{} Loss: {}'.format(epoch+1,i,len(train_loader),loss))
    return

def valid(val_loader, model,criterion):
    #validation definition
    model.eval()
    return_pred, return_tar = [], []
    for i, (input, target) in enumerate(val_loader):
        #target = torch.reshape(target,[target.size(0)])
        with torch.no_grad():
            input_var = Variable(input.cuda(async=True))
            target_var = Variable(target.cuda(async=True))
        output = model(input_var)
        #loss = model.multi_label_sigmoid_cross_entropy_loss(output,target_var)
        #loss = criterion(output, target_var)
        
        return_pred.extend(output.data.cpu().tolist())
        return_tar.extend(target.tolist())
        #print('{}/{} ''acc: {:.4f}'.format(i+1,len(val_loader),acc(output,target)))
    return return_pred, return_tar

def build_model(model_type = 0):
    #build model
    model = alexnet(num_classes=cfg.class_number)
    model.cuda()
    criterion = myloss().cuda()
    #criterion = nn.functional.nll_loss().cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    return model,criterion

def adjust_learning_rate(LR,epoch,freq=cfg.lr_decay_every_epoch):
    if(epoch%freq==0 and epoch!=0):
        LR = LR*cfg.lr_decay_rate
    return LR
    
def main():
    #main function
    Epoch = cfg.epoch
    LR = cfg.lr
    B_size = cfg.train_batch_size
    #test_every_epoch = cfg.test_every_epoch
    #VALID = False
    model_name = ['alex','vgg16','vggf','cc','cct','incep','net','region_alex']
    start = time.time()
    model,criterion = build_model(model_type = cfg.model_type)
    dataset = DataSet(cfg)
    logfile = 'logfile/'+model_name[cfg.model_type]+'_train.txt'
    with open(logfile,'w')as fp:
        fp.close()
    
    print('Start Training!')
    for epoch in range(Epoch):
        LR = adjust_learning_rate(LR,epoch)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        train(dataset.train_loader,model,criterion,optimizer,epoch,logfile)
        if((epoch+1) % cfg.test_every_epoch == 0):
            print('validating......')
            pre,tar = valid(dataset.test_loader,model,criterion)
            pre = np.array(pre)
            tar = np.array(tar)
            np.savetxt('results/'+model_name[cfg.model_type]+str(epoch+1)+'.txt',pre)
            np.savetxt('results/'+model_name[cfg.model_type]+str(epoch+1)+'_tar.txt',tar)
            print('Accuracy: ',acc(pre,tar))
            print('Saving model...')
            torch.save(model.state_dict(),'results/'+model_name[cfg.model_type]+'_ck.pth')
    print('Done!')

if __name__ == '__main__':
    main()
