import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio


class HybridLoss(nn.Module):
    def __init__(self, n_ffts):
        super().__init__()
        self.loss_mss_func = MSSLoss(n_ffts)
        self.f0_loss_func = F0L1Loss()
        self.fo_slow_loss_func = F0SlowLoss()

    def forward(self, y_pred, y_true, f0_pred, f0_true):
        loss_mss = self.loss_mss_func(y_pred, y_true)
        loss_f0 = self.f0_loss_func(f0_pred, f0_true)
        # loss_f0_slow = self.fo_slow_loss_func(f0_pred)
        loss = loss_mss + loss_f0 # + loss_f0_slow

        return loss, (loss_mss, loss_f0) #, loss_f0_slow)


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0.75, eps=1e-7, name='SSSLoss'):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        self.name = name
    def forward(self, x_true, x_pred):
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])
        
        # print('--------')
        # print(min_len)
        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)

        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)
        # print('--------\n\n\n')

        S_true = self.spec(x_true)
        S_pred = self.spec(x_pred)
        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())

        loss = linear_term + self.alpha * log_term
        return {'loss':loss}
        

class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)

    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(self, n_ffts, alpha=1.0, ratio = 1.0, overlap=0.75, eps=1e-7, use_reverb=True, name='MultiScaleLoss'):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        self.ratio = ratio
        self.name = name
    def forward(self, x_pred, x_true, return_spectrogram=True):
        x_pred = x_pred[..., :x_true.shape[-1]]
        if return_spectrogram:
            losses = []
            spec_true = []
            spec_pred = []
            for loss in self.losses:
                loss_dict = loss(x_true, x_pred)
                losses += [loss_dict['loss']]
        
        return self.ratio*sum(losses).sum()


class F0L1Loss(nn.Module):
    """
    crepe loss with pretrained model
    """

    def __init__(self, name = 'F0L1Loss'):
        super().__init__()
        self.iteration = 0
    def forward(self, f0_predict, f0_hz_true):

        # print('pitch pred:', f0_predict[0,:])
        # print('pitch anno:', f0_hz_true[0,:])
        self.iteration += 1
        
        if (len(f0_hz_true.size()) != 3):
            f0_hz_true = f0_hz_true.unsqueeze(-1)
        
        if torch.sum(f0_hz_true>=50) < 10:
            return torch.tensor(0.0)
        if self.iteration > 5000:
            f0_predict = torch.where(f0_hz_true<50, f0_predict*0.0, f0_predict)
            loss = F.l1_loss(torch.log(f0_hz_true+1e-3), torch.log(f0_predict+1e-3), reduction='sum')
            loss = loss / torch.sum(f0_hz_true>=50)
        else:
            loss = F.l1_loss(torch.log(f0_hz_true+1e-3), torch.log(f0_predict+1e-3), reduction='mean')
        return torch.sum(loss)


class F0SlowLoss(nn.Module):
    """
    crepe loss with pretrained model
    """

    def __init__(self, name = 'F0SlowLoss'):
        super().__init__()
        
    def forward(self, f0_predict):
        loss = F.l1_loss(torch.log(f0_predict+1e-3)[:,1:,:], torch.log(f0_predict+1e-3)[:,:-1,:])
        
        return torch.sum(loss)
