# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: yjwu

Python 3.5.2

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import os

from PIL import Image
import numpy as np
import time
#from spiking_model import*
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 7
batch_size  = 1
learning_rate = 1e-3
num_epochs = 5 # max epoch
# define approximate firing function
names = 'spiking_model'
intensity=1
data_path =  './raw/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


array_of_img = []
directory_name='100'
def read_directory():
    for filename in os.listdir(r""+directory_name):
        img = cv2.imread(directory_name + "/" + filename,cv2.IMREAD_GRAYSCALE)
        array_of_img.append(img)
read_directory()
trans=transforms.ToTensor()
for i in range(len(array_of_img)):
    array_of_img[i]=trans(array_of_img[i])

train_loader = torch.utils.data.DataLoader(array_of_img, batch_size=batch_size, shuffle=True, num_workers=0)

#train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

list=[]
i=0
with open('groundtruth.txt') as groudtruth:
    lines=groudtruth.readlines()
    for event in lines:
        i=i+1
        event=event.strip().split()
        event[1]=float(event[1])
        event[2]=float(event[2])
        event[3]=float(event[3])
        event[4]=float(event[4])
        event[5]=float(event[5])
        event[6]=float(event[6])
        event[7]=float(event[7])

        list.append(event)

        if i==2000:
          break

# img =Image.open('fakeimg.png')
# # img.save('tst1.png')
# img=np.array(img)
# # img = Image.fromarray(np.uint8(img))
# totensor=transforms.Compose(
#     [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
# )
#
# img=totensor(img)
# img = img.unsqueeze(0)
# event_label=open('groundtruth.txt')
# label=[0.111111,0.222222,0.333333,0.444444,0.555555,0.666666]
# label=np.array(label)
# label = label.astype(np.float32)
# label=torch.from_numpy(label)


# test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
#acc_record = list([])
# loss_train_record = list([])
# loss_test_record = list([])




class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops, x, mem, spike):

    mem = mem * decay * (1. - spike) + ops(x)

    spike = act_fun(mem) # act_fun : approximation firing function

    return mem, spike

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [[180,240], [90,120], [30,40]]
# fc layer
cfg_fc = [300, 100]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(300* cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3= nn.Linear(100,7)



    def forward(self, input, time_window = 5):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1],180, 240, device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], 90, 120, device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, 32, 30, 40, device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2,x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 3)

            c3_mem,c3_spike=mem_update(self.conv3,x,c3_mem,c3_spike)


            x = F.avg_pool2d(c3_spike, 2)
            x = x.view(batch_size, -1)


            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike


            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)

            h2_sumspike += h2_spike
            predict=self.fc3(h2_sumspike)


        outputs = predict / time_window


        return outputs
snn = SCNN()
snn.to(device)
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

for epoch in range(1):
    running_loss = 0
    start_time = time.time()
    for i, (images) in enumerate(train_loader):
        if i==2000:
            break
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs = snn(images)
        #print(outputs)

        #label=labels.unsqueeze(0)

        #labels_ = torch.zeros(batch_size, 6).scatter_(1, label.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), torch.tensor(list[i][1:8],dtype=float))

        print(outputs)
        #print(loss)


        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    # if (i+1)%100 == 0:
    #         print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
    #             %(epoch+1, num_epochs, i+1, len(train_loader)//batch_size,running_loss ))
    #         running_loss = 0
    #         print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    # with torch.no_grad():
    #     for batch_idx, (inputs) in enumerate(train_loader):
    #         inputs = inputs.to(device)
    #         optimizer.zero_grad()
    #
    #         outputs = snn(inputs)
    #         print(outputs)
    #         labels_ = torch.zeros(batch_size, 10).scatter_(1, list.view(-1, 1), 1)
    #         #print(labels_)
    #         loss = criterion(outputs.cpu(), labels_)
    #         _, predicted = outputs.cpu().max(1)
    #         total += float(targets.size(0))
    #         correct += float(predicted.eq(targets).sum().item())
            # if batch_idx %100 ==0:
            #     acc = 100. * float(correct) / float(total)
            #     print(batch_idx, len(test_loader),' Acc: %.5f' % acc)
    #
    # print('Iters:', epoch,'\n\n\n')
    # print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    # acc = 100. * float(correct) / float(total)
    # acc_record.append(acc)
    # if epoch % 5 == 0:
    #     print(acc)
    #     print('Saving..')
    #     state = {
    #         'net': snn.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #         'acc_record': acc_record,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt' + names + '.t7')
    #     best_acc = acc