# ----------------------------------
# File: LeVoice.py
# ----------------------------------

import torch
from torch import nn


class LeVoice(nn.Module):
    def __init__(self, nfreq):
        super(LeVoice, self).__init__()

        self.nfreq = nfreq
        
        nhid = 64
        nclass = 10

        self.tf = nn.Linear(nfreq, nhid)
        self.gru1 = nn.GRU(nhid, nhid)
        self.gru2 = nn.GRU(nhid, nhid)

        self.output = nn.Sequential(
                        nn.Linear(nhid, nfreq),
                        nn.Linear(nfreq, nclass)
                      )

    def forward(self, feat):
        '''
        Args:
            - feat: <tensor> (N, nFrm, nFreq)
        '''
        feat = feat.permute(1, 0, 2)

        hid1 = self.tf(feat)
        hid2, hn2 = self.gru1(hid1)
        hid3, hn3 = self.gru2(hid2)

        pred = self.output(hid3)
        pred = pred.permute(1, 0, 2)

        return pred


if __name__ == '__main__':
    import pdb

    nfrm  = 100
    nfreq = 40
    batch = 8

    model = LeVoice(nfreq)

    x = torch.rand(batch, nfrm, nfreq)
    pred = model(x)

    pdb.set_trace()
