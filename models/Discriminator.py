# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: Reveal.py
@time: 2018/3/20

"""
import torch
import torch.nn as nn
'''
class UnetRevealMessage(nn.Module):
    def __init__(self):
        super(UnetRevealMessage, self).__init__()

        self.reveal1 = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            #nn.Sigmoid()
        )
        self.reveal12 = nn.Sequential(
            nn.Linear(1024, 1200),
            nn.BatchNorm1d(1200),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1200, 1400),
            nn.BatchNorm1d(1400),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1400, 1600),
            nn.BatchNorm1d(1600),
            nn.Sigmoid()
        )
    def forward(self, input):
        out = self.reveal1(input)
        out = out.view(32, -1)
        out = self.reveal12(out)
        return out
'''

class DiscriminatorNet(nn.Module):
    def __init__(self, nc=3, nhf=8, output_function=nn.Sigmoid):
        super(DiscriminatorNet, self).__init__()
        #self.key_pre = RevealPreKey()
        # input is (3+1) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 4, 2, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            #nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            #nn.BatchNorm2d(nhf*2),
            #nn.ReLU(True),
            #nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            #nn.BatchNorm2d(nhf*4),
            #nn.ReLU(True),
            #nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            #nn.BatchNorm2d(nhf*2),
            #nn.ReLU(True),
            #nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            #nn.BatchNorm2d(nhf),
            #nn.ReLU(True),
            nn.Conv2d(nhf, nc, 4, 2, 1),
            nn.BatchNorm2d(nc),
            nn.ReLU(True)
            #output_function()
        )
        self.linear = nn.Sequential(
            nn.Linear(192, 1),
            output_function()
        )

        # nn.Sigmoid()
        #self.reveal_Message = UnetRevealMessage()

    def forward(self, input):
        #pkey = pkey.view(-1, 1, 32, 32)
        #pkey_feature = self.key_pre(pkey)

        #input_key = torch.cat([input, pkey_feature], dim=1)
        ste_feature = self.main(input)
        out = ste_feature.view(ste_feature.shape[0], -1)

        out = self.linear(out)
        return out
class DiscriminatorNet1(nn.Module):
    def __init__(self):
        super(DiscriminatorNet1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3072, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity