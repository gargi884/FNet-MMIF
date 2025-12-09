import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchprofile import profile_macs
import numpy as np
from typing import List,Any

EPSILON = 1e-4
MAX = 1e2
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def sigmoidal_htheta(x,theta):
    gamma=100
    alpha=0.1
    return torch.sign(x)*(torch.abs(x)-alpha*theta)/(1+torch.exp(-gamma*(torch.abs(x)-theta)))    
    
class basic_layer(nn.Module):
    def __init__(self,num_channel=1,num_filter=64,kernel_size=9):
        super(basic_layer, self).__init__()
        self.ch_down = nn.Conv2d(in_channels=num_filter, out_channels=num_channel, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.ch_down.weight.data)
        self.ch_up = nn.Conv2d(in_channels=num_channel, out_channels=num_filter, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.ch_up.weight.data)
    def forward(self, x,rho):
        x1 = self.ch_down(x)
        x2 = self.ch_up(x1)
        return rho*(x-x2)
    
class x0_block(nn.Module):
    def __init__(self,num_channel=1,num_filter=64,kernel_size=9):
        super(x0_block, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=num_channel, out_channels=num_filter, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_in.weight.data)
    def forward(self, x,theta):
        return sigmoidal_htheta(self.conv_in(x),theta)        
    
class x1_block(nn.Module):
    def __init__(self,num_channel=1,num_filter=64,kernel_size=9):
        super(x1_block, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=num_channel, out_channels=num_filter, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_in.weight.data)
        self.basic_layer = basic_layer(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size)
        
    def forward(self, xt_1,e,theta,ro):
        xt_h = self.basic_layer(xt_1,1+ro)+self.conv_in(e)
        return sigmoidal_htheta(xt_h,theta) 
    
class IM(nn.Module):
    def __init__(self,num_channel=1,num_filter=64,kernel_size=9):
        super(IM, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=num_channel, out_channels=num_filter, kernel_size=kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_in.weight.data)
        self.basic_layer1 = basic_layer(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size)
        self.basic_layer2 = basic_layer(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size)
        
    def forward(self, xt_1,xt_2,e,theta,ro):
        xt_1p = self.basic_layer1(xt_1,1+ro)
        xt_2p = self.basic_layer2(xt_2,ro)
        xt_h = xt_1p-xt_2p+self.conv_in(e)
        return sigmoidal_htheta(xt_h,theta) 
    
class LZSC(nn.Module):
    def __init__(self,num_channel=1,num_filter=64,kernel_size=9,n_layer:int=4):
        super(LZSC, self).__init__()
        self.n_layer:int = n_layer
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        self.w_ro = nn.Parameter(torch.Tensor([0.5]))
        self.b_ro = nn.Parameter(torch.Tensor([0]))
        self.x0_block = x0_block(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size)
        self.x1_block = x1_block(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size)
        self.IM: List[Any] = []
        count = 0
        for i in range(self.n_layer-1):
            count += 1
            self.IM.append(IM(num_channel=num_channel,num_filter=num_filter,kernel_size=kernel_size))
        self.IM = nn.ModuleList(self.IM)
        self.Sp = nn.Softplus()
    
    def forward(self, e):
        theta0 = self.Sp(self.b_theta)
        x0 = self.x0_block(e,theta0)
        theta1 = self.Sp(self.w_theta+self.b_theta)
        ro1 = (self.Sp(self.w_ro+self.b_ro)-self.Sp(self.b_ro))/self.Sp(self.w_ro+self.b_ro)
        x1 = self.x1_block(x0,e,theta1,ro1)
        xk_1=x0
        xk=x1
        for index, v in enumerate(self.IM):
            k = index+2
            thetak = self.Sp(self.w_theta*k+self.b_theta)
            rok = (self.Sp(self.w_ro*k+self.b_ro)-self.Sp(self.b_ro))/self.Sp(self.w_ro*k+self.b_ro)
            xkt = v(xk, xk_1, e,thetak,rok)
            xk_1 = xk
            xk = xkt
        return xk
        
        
class IFNet(nn.Module):
    def __init__(self,channel = 1,num_filters = 64,kernel_size = 9):
        super(IFNet, self).__init__()
        self.channel = channel
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.net_x1 = LZSC(num_channel=self.channel)
        self.conv_x1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_x1.weight.data)
        self.net_x2 = LZSC(num_channel=self.channel)
        self.conv_x2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding='same', bias=False)
        nn.init.xavier_uniform_(self.conv_x2.weight.data)
    def forward(self, f):
        x1 = self.net_x1(f)
        x2 = self.net_x2(f)
        I1 = self.conv_x1(x1)
        I2 = self.conv_x1(x2)
        return I1,I2
        
if __name__ == '__main__':
    model = IFNet().to(device)
    x = torch.randn((1, 1, 128, 128)).to(device)
    I1,I2 = model(x)
    print(I1.shape)
    # flops = profile_macs(model, inputs)
    # print('Model flops: ',flops/10**9,'G')
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model parameters: ',params_num/10**3,'K')
    
    script_model = torch.jit.script(model)
    script_model.save('model_scripted.pt')
    
    model1 = torch.jit.load('model_scripted.pt')
    I1,I2 = model1(x)
    print(I1.shape)
    
    # run_times=[]
    # for i in range(1000):
        # start_time=time.time()
        # x = model(inputs)
        # end_time=time.time()
        # run_times.append(end_time-start_time)
        
    # run_times=np.array(run_times)
    # print('Average runtime: ', np.mean(run_times),'s')