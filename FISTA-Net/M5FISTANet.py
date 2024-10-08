# -*- coding: utf-8 -*-
"""
Created on June 17, 2020

ISTANet(shared network with 4 conv + ReLU) + regularized hyperparameters softplus(w*x + b). 
The Intention is to make gradient step \mu and thresholding value \theta positive and monotonically decrease.

@author: XIANG
"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt


layer = 1


def test_plot(x, file_name):
    #plt.imshow(x[:,:,:64,:].squeeze().detach().numpy())
    #plt.savefig(file_name)
    pass

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


# define basic block of FISTA-Net
class BasicBlock(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, features=32):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

    def forward(self, x, PhiTPhi, PhiTb, LTL, mask, lambda_step, soft_thr, epoch):
        global layer
        # convert data format from (batch_size, channel, row, col) to (batch_size, channel, row, col)
        # x = torch.squeeze(x, 1)
        # x = mask.mm(x)   - nekünk a teljes kép kell

        # naive gradient descent update
        x = x - self.Sp(lambda_step) * (torch.matmul(PhiTPhi, x) - PhiTb)
        #if not os.path.exists('C:\\Users\\Kovács Ottó\\Documents\\GitHub\\ThermDataGen\\FistaNet\\FISTA-Net\\testing\\epoch_%d' % epoch):
        #    os.makedirs('C:\\Users\\Kovács Ottó\\Documents\\GitHub\\ThermDataGen\\FistaNet\\FISTA-Net\\testing\\epoch_%d' % epoch)
        #file_name = '.\\testing\\epoch_%d\\/8a_layer%d.png' % (epoch, layer)
        #test_plot(x, file_name)

        # quadratic tv gradient descent from doi:  10.1109/TMI.2009.2022540 Eq. (10)
        # TODO: az inverzet cikluson kívűl kiszámolni
        # CIKK: iteráció kék oldala - gradient descent module --\/
        # x = x - self.Sp(lambda_step) * torch.inverse(PhiTPhi + 0.001 * LTL).mm(PhiTPhi.mm(x) - PhiTb - 0.001 * LTL.mm(x))

        # convert (batch_size, channel, row, col) to (batch_size, channel, row, col)
        # x = torch.mm(mask.t(), x)
        # x = x.view(pnum, pnum, -1)
        # x = x.unsqueeze(1)
        # x_input = x.permute(3, 0, 1, 2)
        x_input = x

        # CIKK: minden, ami ez alatt van a narancssárga rész - proximal mapping module --\/
        x_D = self.conv_D(x_input.float())

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        x = F.relu(x)
        x = self.conv3_forward(x)
        x = F.relu(x)
        x_forward = self.conv4_forward(x)

        # soft-thresholding block
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

        x = self.conv1_backward(x_st)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_backward = self.conv4_backward(x)

        x_G = self.conv_G(x_backward)

        # prediction output (skip connection); non-negative output
        x_pred = F.relu(x_input + x_G)

        # compute symmetry loss
        x = self.conv1_backward(x_forward)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_D_est = self.conv4_backward(x)
        symloss = x_D_est - x_D

        return [x_pred, symloss, x_st]


class FISTANet(nn.Module):
    def __init__(self, LayerNo, featureNo, Phi, L, mask):
        super(FISTANet, self).__init__()
        self.LayerNo = LayerNo
        self.Phi = Phi
        self.L = L
        self.mask = mask
        onelayer = []

        self.bb = BasicBlock(features=featureNo)
        for i in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

    def forward(self, x0, b, epoch):
        """
        Phi   : system matrix; default dim 104 * 3228;
        mask  : mask matrix, dim 3228 * 4096
        b     : measured signal vector;
        x0    : initialized x with Laplacian Reg.
        """
        global layer

        if layer == 1:
            if not os.path.exists('.\\testing'):
                os.makedirs('.\\testing')
            file_name = '.\\testing\\00_initial_x.png'
            test_plot(x0, file_name)
            file_name = '.\\testing\\00_initial_b.png'
            test_plot(b, file_name)

        # convert data format from (batch_size, channel, vector_row, vector_col) to (vector_row, batch_size)
        # NEW: in our case it is (batch_size, row, col)
        # b = torch.squeeze(b, 1)

        PhiTPhi = torch.matmul(self.Phi.t(), self.Phi)
        PhiTb = torch.matmul(self.Phi.t(), b)
        LTL = [] #self.L.t().mm(self.L)

        # initialize the result
        xold = x0
        y = xold
        layers_sym = []     # for computing symmetric loss
        layers_st = []      # for computing sparsity constraint
        xnews = []          # iteration result
        xnews.append(xold)

        for i in range(self.LayerNo):
            # CIKK: (15) --\/
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu
            # CIKK: (8a) + (8b) - nagy iteráció rész --\/
            [xnew, layer_sym, layer_st] = self.fcs[i](y, PhiTPhi, PhiTb, LTL, self.mask, mu_, theta_, epoch)
            file_name = 'C:\\Users\\Kovács Ottó\\Documents\\GitHub\\ThermDataGen\\FistaNet\\FISTA-Net\\testing\\epoch_%d\\8b_layer%d.png' % (epoch, layer)
            test_plot(xnew, file_name)

            # CIKK: - rho update rész --\/
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            # CIKK: (8c) - következő réteg bemenetének számolása --\/
            y = xnew + rho_ * (xnew - xold)  # two-step update
            file_name = 'C:\\Users\\Kovács Ottó\\Documents\\GitHub\\ThermDataGen\\FistaNet\\FISTA-Net\\testing\\epoch_%d\\8c_layer%d.png' % (epoch, layer)
            test_plot(y, file_name)
            layer += 1

            xnews.append(xnew)   # iteration result
            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        return [xnew, layers_sym, layers_st]
