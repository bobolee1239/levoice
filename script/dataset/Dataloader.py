# ----------------------------------
# File: Dataloader.py
# ----------------------------------
import sys 
if '..' not in sys.path:
    sys.path.append('..')

from torch.utils.data            import DataLoader, ConcatDataset
from dataset.VoiceCommandDataset import VoiceCommandDataset

import config

TRAIN_CMD_FOLDER = config.TRAIN_CMD_FOLDER
TRAIN_BGN_FOLDER = config.TRAIN_BGN_FOLDER
CLASS_TABLE_FILE = config.CLASS_TABLE_FILE

# -------------------------------------------------

def get_train_dataloader(batch_size=32):
    bgnfolder = TRAIN_BGN_FOLDER[0]
    train_cmd_sets = [VoiceCommandDataset(d, bgnfolder, CLASS_TABLE_FILE) for d in TRAIN_CMD_FOLDER]
    train_cmd_set = ConcatDataset(train_cmd_sets)

    loader = DataLoader(
                train_cmd_set, 
                batch_size=batch_size, 
                shuffle=True
             )
    return loader


# -----------------------------------------------

if __name__ == '__main__':
    import pdb
    import numpy as np
    import librosa
    import matplotlib.pyplot as plt

    batch_size = 9

    loader = get_train_dataloader(batch_size)

    def plot_feature1(feat):
        import librosa.display
        fig, ax = plt.subplots()
        S_dB = 10.0 * np.log10(feat + 1e-12)
        img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=16000,
                         fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')

    def plot_feature2(feat, gamma=0.2):
        cmap = 'inferno'
        fig, ax = plt.subplots()
        S_dB = feat ** gamma
        ax.imshow(S_dB, origin='lower', cmap=cmap)
        ax.set(title='Mel-frequency spectrogram')

    for n, batch in enumerate(loader):
        if n != 0:
            break
            
        sigs, specs, mel_specs, label = batch

        for b in range(batch_size):
            mel_spec = mel_specs[b].permute(1, 0).numpy()
            spec = np.abs(specs[b].permute(1, 0).numpy()) ** 2
            plot_feature2(mel_spec)
            plot_feature2(spec)

            pdb.set_trace()