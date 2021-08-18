# ----------------------------------
# File: train_LeVoice.py
# ----------------------------------
import os
import sys
import json
import torch
import logging
import librosa
import numpy       as np

if '..' not in sys.path:
    sys.path.append('..')

from model.LeVoiceFNN      import LeVoice

import config
# ---------------------------------------------------------------
SR       = config.SR
NFFT     = config.NFFT
NMEL     = config.NMEL
HOP_SIZE = config.HOP_SIZE
WIN_SIZE = config.WIN_SIZE
WIN_TYPE = config.WIN_TYPE

LABEL_ELSE = config.N_CLASS - 1

logger = logging.getLogger('LeVoice')
logging.basicConfig(level=logging.INFO)
# ---------------------------------------------------------------
def load_model(net, path):
    logger.info(f'* Loading LeVoice model: {path}')
    net.load_state_dict(
        torch.load(path,
                   map_location=torch.device('cpu'))
    )


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
    table_file = args.table
    
    class_table = None
    if table_file:
        with open(table_file, 'r') as fd:
            class_table = json.load(fd)

    nfreq = NMEL
    model = LeVoice(nfreq)

    load_model(model, model_path)

    sig, sr = audioread_mono(audio)

    feat = extract_feature(sig)
    with torch.no_grad():
        output = model(feat)

    # (1, nFrm, nClass)
    _, pred = torch.max(output, 2)
    pred = pred.squeeze()

    #result = np.where(pred == 10, 0, pred)
    pred_stat = np.bincount(pred[pred != LABEL_ELSE])
    prediction = int(np.argmax(pred_stat))
    
    print(f'Predict Sequence    :')
    print(f'{pred}')
    print(f'Prediction Statistic: {int(prediction)}')
    #print(f'Prediction          : {int(pred_stat)}')

    if class_table is not None:
        print('---' * 20)
        print(f'Prediction Class: {class_table[str(prediction)]}')



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
    parser.add_argument(
        '-t',
        '--table', 
        type=str,
        required=False,
        help='Class table to look up'
    )

    args = parser.parse_args()

    main(args)
