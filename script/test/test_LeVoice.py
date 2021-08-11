# ----------------------------------
# File: train_LeVoice.py
# ----------------------------------
import os
import sys
import torch
import librosa
import numpy       as np

if '..' not in sys.path:
    sys.path.append('..')

from model.LeVoice      import LeVoice

SR       = 16000
NFFT     = 1024
NMEL     = 40
HOP_SIZE = 160
WIN_SIZE = 640
WIN_TYPE = 'hann'
# ---------------------------------------------------------------
# ---------------------------------------------------------------
def load_model(net, path):
    net.load_state_dict(torch.load(path))


def audioread_mono(wavfile):
    sig, sr = librosa.load(wavfile, sr=SR, dtype=np.float32)

    if len(sig.shape) > 1:
        sig = sig[:, 0]

    return sig, sr


def extract_feature(sig):
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
    return torch.tensor(mel_spectra).permute(1, 0).unsqueeze(0)

def main(args):
    audio      = args.audio
    model_path = args.model

    nfreq = 40
    model = LeVoice(nfreq)

    load_model(model, model_path)

    sig, sr = audioread_mono(audio)

    feat = extract_feature(sig)
    with torch.no_grad():
        output = model(feat)

    _, pred = torch.max(output, 1)

    print(f'Predicted Class: {pred.item()}')
    print(f'Model Output: {output}')


# ----------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'audio',
        type=str,
        help='Test Audio File'
    )
    parser.add_argument(
        '-m',
        '--model', 
        type=str,
        required=True,
        help='NN model to be loaded'
    )

    args = parser.parse_args()

    main(args)