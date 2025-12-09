import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class loss_fusion(nn.Module):
    def __init__(self,coeff_int=1,coeff_grad=1):
        super(loss_fusion, self).__init__()
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad

    def forward(self,pre,target):
        loss_int=F.l1_loss(pre,target)
        loss_grad=F.l1_loss(kornia.filters.SpatialGradient()(pre),kornia.filters.SpatialGradient()(target))
        loss_total=self.coeff_int*loss_int+self.coeff_grad*loss_grad#+Loss_SSIM
        return loss_total