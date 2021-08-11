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

SR       = 16000
NFFT     = 1024
NMEL     = 40
HOP_SIZE = 160
WIN_SIZE = 640
WIN_TYPE = 'hann'

SECONDS  = 1
LENGTH   = SECONDS * SR
# -----------------------------------------------------
def pad_to(sig, target_len):
    length = sig.shape[0]

    if length >= target_len:
        return sig[:target_len]
    
    pad_len = int(target_len - length)

    return np.concatenate((sig, np.zeros((pad_len, ))))


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
                center=False
              )
    return spectra


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
        label = int(os.path.basename(wavfile).split('_')[0])
        sig, sr = audioread_mono(wavfile)

        sig = pad_to(sig, LENGTH)

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
                         center=False, 
                         power=2.0
                       )
        
        # (nFrm, nFreq)
        spectrum    = np.transpose(spectrum).astype(np.complex64)
        mel_spectra = np.transpose(mel_spectra).astype(np.float32)
        return sig, spectrum, mel_spectra, label
