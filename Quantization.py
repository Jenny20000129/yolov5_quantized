import collections
import torch.nn.functional as F
import os
import math
import random
import torch
import torch.nn as nn
import numpy as np

from mish_cuda import MishCuda as Mish


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit):
        if bit == 0:
            act = x * 0
        else:
            step = 2 ** (bit) - 1
            act = torch.round(x * step) / step

            # S = torch.max(torch.abs(x))
            # if S==0:
            #     act = x*0
            # else:
            #     step = 2 ** (bit)-1
            #     R = torch.round(torch.abs(x) * step / S)/step
            #     act =  S * R * torch.sign(x)
        return act

    @staticmethod
    def backward(ctx, g):
        return g, None


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()
        self.qbit = 8
        self.abit = 8
        # HYQ CLOSE W_QUANT
        # self.alphq = torch.tensor(0.0)
        self.alphq = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.alpha = torch.tensor(0.0)

    def W_Quant(self, weight):
        wq = STE.apply(weight, self.qbit)
        if self.alphq > 0:
            dq = STE.apply(weight, self.qbit + 1)
        else:
            dq = STE.apply(weight, self.qbit - 1)
        return torch.abs(
            self.alphq) * dq + (1 - torch.abs(self.alphq)) * wq

    def A_Quant(self, mod):
        aq = STE.apply(mod, self.abit)
        if self.alpha > 0:
            cq = STE.apply(mod, self.abit + 1)
        else:
            cq = STE.apply(mod, self.abit - 1)
        return torch.abs(self.alpha) * cq + (1 - torch.abs(self.alpha)) * aq

    def init_quant(self, qbit, abit):
        self.qbit = qbit
        self.abit = abit

    def quant(self, epoch):  # HYQ ADD EPOCH
        self.qbit += torch.round(self.alphq)
        self.abit += torch.round(self.alpha)
        # if epoch >= 120:
        #     max_bit = 4
        # elif epoch >= 90:
        #     max_bit = 5
        # elif epoch >= 30:
        #     max_bit = 6
        # else:
        #     max_bit = 8

        max_bit = 4
        # if epoch >= 30:
        #     max_bit = 4
        # else:
        #     max_bit = 5
        self.qbit = max(min(self.qbit, max_bit), 2)  # max_bit原为8
        self.abit = max(min(self.abit, max_bit), 2)  # max_bit原为8
        alphqd = self.alphq.data
        self.alphq.data = alphqd - torch.round(alphqd)
        alphad = self.alpha.data
        self.alpha.data = alphad - torch.round(alphad)
        print(self.alpha.data, self.alphq.data)
        return self.qbit, self.abit, 0, 0

    def quantloss_Conv(self, bo, c2):
        # （平均weight比特-目标weight比特）*alphq + 0.5*(平均activation比特-目标a比特)*alpha
        # activation平均比特要高于weight平均比特，a更新也慢于w

        wl = torch.abs(self.qbit + self.alphq - 3.5) / np.sqrt(bo)  # 原3.5
        al = torch.abs(self.abit + self.alpha - 3.5) / np.sqrt(c2)  # 原4
        # qloss = self.qbit+self.alphq+self.alpha+self.abit  # 加权为1
        # bitops=(self.qbit+self.alphq)*(self.abit+self.alpha)*self.bo  # bitops为导向，还需乘以这一层输出的特征图尺寸h*w
        # wloss = torch.abs(self.qbit+self.alphq-2.6)
        # aloss = 0.5*torch.abs(self.abit+self.alpha-3.6)
        return wl, al
        # return bitops

    def quantloss_BottleneckCSP(self, cin, co):
        # （平均weight比特-目标weight比特）*alphq + 0.5*(平均activation比特-目标a比特)*alpha
        # activation平均比特要高于weight平均比特，a更新也慢于w
        # bitops2 = (self.cv2qbit+self.alphq2) * (self.cv2abit+self.alpha2) * self.bo2
        # bitops3 = (self.cv3qbit+self.alphq3)*(self.cv3abit+self.alpha3)*self.bo3
        wl = torch.abs(self.qbit + self.alphq - 3.5) / np.sqrt(cin * co) + \
            torch.abs(self.qbit + self.alphq - 3.5) / np.sqrt(co * co)
        al = torch.abs(self.abit + self.alpha - 4) + \
            torch.abs(self.abit + self.alpha - 4)
        al = al / np.sqrt(co)
        return wl, al

    def quantloss_BottleneckCSP2(self, bo):
        # qloss = self.qbit + self.alphq + self.alpha + self.abit  # 加权为1
        # return qloss
        # bitops = (self.qbit + self.alphq) * (self.abit + self.alpha) * self.bo * self.bo # bitops为导向
        # return bitops
        wl = torch.abs(self.qbit + self.alphq - 3.5) / np.sqrt(bo * bo)
        al = torch.abs(self.abit + self.alpha - 4) / np.sqrt(bo)
        return wl, al

    def quantloss_SPPCSP(self, c1, co):
        # qloss = self.qbit + self.alphq + self.alpha + self.abit  # 加权为1
        # return qloss
        # bitops = (self.qbit + self.alphq) * (self.abit + self.alpha) * self.c1 * self.co  # bitops为导向
        # return bitops
        # （平均weight比特-目标weight比特）*alphq + 0.5*(平均activation比特-目标a比特)*alpha
        # activation平均比特要高于weight平均比特，a更新也慢于w
        wl = torch.abs(self.qbit + self.alphq - 3.5) / np.sqrt(c1 * co)
        al = torch.abs(self.abit + self.alpha - 4) / np.sqrt(co)
        # wloss = torch.abs(self.qbit+self.alphq-2.6)
        # aloss = 0.5*torch.abs(self.abit+self.alpha-3.6)
        return wl, al
