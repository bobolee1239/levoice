# ----------------------------------
# File: VoiceCommandDataset.py
# ----------------------------------
import os 
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
# -----------------------------------------------------
def pad_to(sig, label, target_len):
    length = sig.shape[0]

    if length >= target_len:
        sig_hat   = sig[:target_len]
        label_hat = np.ones((target_len, )) * label
    else: 
        pad_len = int(target_len - length)
        n_front = random.randint(0, pad_len)
        n_back  = pad_len - n_front

        pad_f  = np.zeros((n_front, ))
        pad_b  = np.zeros((n_back, ))
        labels = np.ones((length, )) * label

        sig_hat   = np.concatenate((pad_f, sig   , pad_b))
        label_hat = np.concatenate((pad_f, labels, pad_b))

    return sig_hat, label_hat

def label_smpl2frm(label_smpl, label_idx):
    label_frm = label_smpl.reshape((-1, HOP_SIZE))
    msk = (np.sum(label_frm, axis=1) > 0)
    return np.where(msk, label_idx, 10)
    

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
    def __init__(self, folder):
        wavs = list(Path(folder).rglob('*.wav'))
        wavs = [str(wav) for wav in wavs]

        self.wavs   = wavs
        self.length = len(wavs)

        random.shuffle(self.wavs)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        wavfile = self.wavs[idx]
        label_id = int(os.path.basename(wavfile).split('_')[0])
        sig, sr = audioread_mono(wavfile)

        sig, label_smpl = pad_to(sig, label_id, LENGTH)

        label = label_smpl2frm(label_smpl, label_id)

        spectrum = stft(sig)
        spectra  = np.abs(spectrum) ** 2.0

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
        # (nFrm, nFreq)
        spectrum    = np.transpose(spectrum).astype(np.complex64)
        mel_spectra = np.transpose(mel_spectra).astype(np.float32)
        label       = label.astype(np.int)

        return sig, spectrum, mel_spectra, label
