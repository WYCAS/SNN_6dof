# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
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
batch_size  = 5
learning_rate = 1e-3
num_epochs = 5 # max epoch
# define approximate firing function
names = 'spiking_model'
intensity=1
data_path =  './raw/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_train=1500

array_of_img_train = []
array_of_img_test = []
directory_name='100'
def read_directory():
    i=0
    for filename in os.listdir(r""+directory_name):
        img = cv2.imread(directory_name + "/" + filename,cv2.IMREAD_GRAYSCALE)
        if i<num_train:
            array_of_img_train.append(img)
        else:
            array_of_img_test.append(img)
        i=i+1
read_directory()
trans=transforms.ToTensor()
for i in range(len(array_of_img_train)):
    array_of_img_train[i]=trans(array_of_img_train[i])

train_loader = torch.utils.data.DataLoader(array_of_img_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader=torch.utils.data.DataLoader(array_of_img_test, batch_size=batch_size, shuffle=True, num_workers=0)

#train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

list_train_y=[]
list_test_y=[]
i=0
with open('groundtruth.txt') as groudtruth:
    lines=groudtruth.readlines()
    for event in lines:
        event=event.strip().split()
        event[1]=float(event[1])
        event[2]=float(event[2])
        event[3]=float(event[3])
        event[4]=float(event[4])
        event[5]=float(event[5])
        event[6]=float(event[6])
        event[7]=float(event[7])
        if i<num_train:
            list_train_y.append(event)
        else:
            list_test_y.append(event)
        i = i + 1

acc_record = list([])
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


        self.fc1 = nn.Linear(300* 32, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3= nn.Linear(100,7)

    def forward(self, input, time_window = 8):
        c1_mem = c1_spike = torch.zeros(batch_size, 1,180, 240, device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, 32, 90, 120, device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, 32, 30, 40, device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, 300, device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, 100, device=device)

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

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images) in tqdm(enumerate(train_loader)):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs = snn(images)
        #print(outputs)
        #label=labels.unsqueeze(0)
        loss = criterion(outputs.cpu(), torch.tensor(list_train_y[i][1:8],dtype=float))

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                %(epoch+1, num_epochs, i+1, len(train_loader)//batch_size,running_loss ))
            running_loss = 0
            print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

