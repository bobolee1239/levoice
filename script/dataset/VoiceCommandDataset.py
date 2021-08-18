# ----------------------------------
# File: VoiceCommandDataset.py
# ----------------------------------
import os 
import json
import random
import torch
import librosa

import numpy     as np 
import soundfile as sf

from pathlib          import Path
from torch.utils.data import Dataset

import config

SR       = config.SR
NFFT     = config.NFFT
NMEL     = config.NMEL
HOP_SIZE = config.HOP_SIZE
WIN_SIZE = config.WIN_SIZE
WIN_TYPE = config.WIN_TYPE

SECONDS  = config.SECONDS
LENGTH   = config.LENGTH

LABEL_ELSE = config.N_CLASS - 1

EPS = np.finfo(float).eps
# -----------------------------------------------------
def pad_to(sig, label, target_len):
    length = sig.shape[0]

    if length >= target_len:
        sig_hat   = sig[:target_len]
        label_hat = np.ones((target_len, )) * label
    else: 
        pad_len = int(target_len - length)
        n_back  = random.randint(8000, pad_len)
        n_front = pad_len - n_back

        pad_f  = np.zeros((n_front, ))
        pad_b  = np.zeros((n_back, ))
        
        labels = np.ones((length, )) * -1

        label_f = np.ones(pad_f.shape) * -1
        label_b1 = np.ones((8000, )) * label
        label_b2 = np.ones((n_back - 8000, )) * -1
        label_b = np.concatenate((label_b1, label_b2), axis=0)

        sig_hat   = np.concatenate((pad_f  , sig   , pad_b  ))
        label_hat = np.concatenate((label_f, labels, label_b))

    return sig_hat, label_hat


def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio


def tailor_bgn(bgn, target_len):
    length = bgn.shape[0]
    
    if length > target_len:
        src = random.randint(0, length - target_len)
        return bgn[src:src+target_len]
    
    bgn2 = np.concatenate((bgn, bgn), axis=0)
    return tailor_bgn(bgn2, target_len)


def snr_mix(voc, _bgn, snr, spl, rms_voc=None):
    bgn = tailor_bgn(_bgn, target_len=voc.shape[0])

    if rms_voc is None:
        rms_voc = (voc ** 2).mean() ** 0.5
    rms_bgn = (bgn ** 2).mean() ** 0.5

    gain_bgn = rms_voc / (10.0 ** (snr / 20.0)) / (rms_bgn + EPS)

    sig = voc + gain_bgn * bgn
    return normalize(sig, target_level=spl)


def label_smpl2frm(label_smpl, label_idx):
    label_frm = label_smpl.reshape((-1, HOP_SIZE))
    msk = (np.sum(label_frm, axis=1) > -160)
    return np.where(msk, label_idx, LABEL_ELSE)
    

def audioread_mono(wavfile):
    sig, sr = librosa.load(wavfile, sr=SR, dtype=np.float32)

    if len(sig.shape) > 1:
        sig = sig[:, 0]

    return sig, sr


def stft(sig):
    spectra = librosa.stft(
                sig, 
                n_fft=NFFT, 
                hop_length=HOP_SIZE, 
                win_length=WIN_SIZE, 
                window=WIN_TYPE, 
                center=True
              )
    return spectra[:, :-1]


class VoiceCommandDataset(Dataset):
    def __init__(self, wavfolder, bgnfolder, class_tablefile):
        super().__init__()

        wavs = list(Path(wavfolder).rglob('*.wav'))
        wavs = [str(wav) for wav in wavs]

        bgns = list(Path(bgnfolder).rglob('*.wav'))
        bgns = [str(bgn) for bgn in bgns]

        class_table = None
        with open(class_tablefile, 'r') as fd:
            class_table = json.load(fd)

        cmds = []
        for wav in wavs:
            class_descrip = os.path.basename(os.path.dirname(wav)).rstrip().lstrip()
            class_id = class_table[class_descrip]
            cmds.append({
                'wav'  : wav,
                'label': class_id
            })

        self.bgns   = bgns
        self.cmds   = cmds
        self.length = len(self.cmds)
        
        # For Augmentation 
        self.silence_prob = 0.1
        self.snr_range    = [0, 40]
        self.spl_range    = [-45, -6]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        wavfile  = self.cmds[idx]['wav'  ]
        label_id = self.cmds[idx]['label'] 

        sig, sr = audioread_mono(wavfile)
        rms_sig = (sig ** 2).mean() ** 0.5

        sig, label_smpl = pad_to(sig, label_id, LENGTH)

        label = label_smpl2frm(label_smpl, label_id)

        # --- Augmentation ---
        if random.random() > self.silence_prob:
            nsefile = random.choice(self.bgns)
            nse, _  = audioread_mono(nsefile)
            snr     = random.randint(*self.snr_range)
            spl     = random.randint(*self.spl_range)

            sig = snr_mix(sig, nse, snr, spl, rms_voc=rms_sig)

        # Pre-emphasized
        alpha = 0.97
        sig = np.append(sig[0], sig[1:] - alpha*sig[:-1])
        spectrum = stft(sig)
        # spectra  = np.abs(spectrum) ** 2.0

        mel_spectra = librosa.feature.melspectrogram(
                        #  S=spectra,
                         y=sig,
                         n_mels=NMEL,
                         sr=SR, 
                         fmax=SR/2,
                         n_fft=NFFT, 
                         hop_length=HOP_SIZE, 
                         win_length=WIN_SIZE, 
                         window=WIN_TYPE, 
                         center=True, 
                         power=2.0
                       )
        mel_spectra = mel_spectra[:, :-1] 
        ## Subtract the mean of coefficient from all frames
        # mel_spectra -= (np.mean(mel_spectra, axis=0) + 1e-8)
        # (nFrm, nFreq)
        spectrum    = np.transpose(spectrum).astype(np.complex64)
        mel_spectra = np.transpose(mel_spectra).astype(np.float32)
        label       = label.astype(np.int)


        return sig, spectrum, mel_spectra, label
