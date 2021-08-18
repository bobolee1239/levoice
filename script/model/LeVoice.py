# ----------------------------------
# File: LeVoice.py
# ----------------------------------
import sys 
if '..' not in sys.path:
    sys.path.append('..')

import torch
import torch.nn.functional as F
import config

from   torch    import nn

# -----------------------------------------

def pad_freq(x, padding):
    '''
    Args:
    --------
        - x: (N, ch, nFrm, nFreq)
        - padding: (freq_low, freq_up)
    '''
    return F.pad(x, padding, "constant", 0) 

def pad_time(x, padding):
    '''
    Args:
    --------
        - x: (N, ch, nFrm, nFreq)
        - padding: (time_left, time_right)
    '''
    return F.pad(x, (0, 0, *padding), "constant", 0) 


class LeVoice(nn.Module):
    def __init__(self, nfreq):
        super(LeVoice, self).__init__()

        self.nfreq = nfreq
        
        nclass = config.N_CLASS
        nhid = 128

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 4, (3, 3))
        self.bn1   = nn.BatchNorm2d(4)

        self.conv2_dep = nn.Conv2d(4, 8, (3, 3), groups=4)
        self.conv2_pt  = nn.Conv2d(8, 8, (1, 1))
        self.bn2       = nn.BatchNorm2d(8)

        self.conv3_dep = nn.Conv2d( 8, 16, (3, 1), groups=8)
        self.conv3_pt  = nn.Conv2d(16, 16, (1, 1))
        self.bn3       = nn.BatchNorm2d(16)

        self.conv4_dep = nn.Conv2d(16, 32, (3, 1), groups=16)
        self.conv4_pt  = nn.Conv2d(32, 32, (1, 1))
        self.bn4       = nn.BatchNorm2d(32)

        self.pre_gru_pt = nn.Conv2d(32, 8, (1, 1))
        self.tf         = nn.Linear(40*8, nhid)

        self.gru1 = nn.GRU(nhid, nhid)
        self.gru2 = nn.GRU(nhid, nhid)

        self.output = nn.Sequential(
                        nn.Linear(nhid, nclass)
                      )

    def forward(self, spectra):
        '''
        Args:
            - feat: <tensor> (N, nFrm, nFreq)
        '''
        # (N, 1, nFrm, nFreq)
        spectra = spectra.unsqueeze(1)
        spectra = pad_time(spectra, (2, 6))
        spectra = pad_freq(spectra, (2, 2))

        spec_hid1 = self.conv1(spectra)
        spec_hid1 = self.bn1(spec_hid1)
        spec_hid1 = self.relu(spec_hid1)

        spec_hid2 = self.conv2_dep(spec_hid1)
        spec_hid2 = self.conv2_pt(spec_hid2)
        spec_hid2 = self.bn2(spec_hid2)
        spec_hid2 = self.relu(spec_hid2)

        spec_hid3 = self.conv3_dep(spec_hid2)
        spec_hid3 = self.conv3_pt(spec_hid3)
        spec_hid3 = self.bn3(spec_hid3)
        spec_hid3 = self.relu(spec_hid3)

        spec_hid4 = self.conv4_dep(spec_hid3)
        spec_hid4 = self.conv4_pt(spec_hid4)
        spec_hid4 = self.bn4(spec_hid4)
        spec_hid4 = self.relu(spec_hid4)

        # (N, 8, nFrm, nFreq)
        spec_hid5 = self.pre_gru_pt(spec_hid4)
        N, nCh, nFrm, nFreq = spec_hid5.shape 
        # (nFrm, N, nFreq)
        feat = spec_hid5.permute(2, 0, 1, 3)
        feat = feat.reshape((nFrm, N, nCh*nFreq))
        hid1 = self.tf(feat)

        hid2, hn2 = self.gru1(hid1)
        hid3, hn3 = self.gru2(hid2)

        hid4 = 0.5 * (hid2 + hid3)
        pred = self.output(hid4)
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
