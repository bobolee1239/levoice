# ----------------------------------
# File: Dataloader.py
# ----------------------------------

from torch.utils.data    import DataLoader, ConcatDataset
from VoiceCommandDataset import VoiceCommandDataset

TRAIN_SET_FOLDER = [
    '/Users/brian/brian_ws/ASR/dataset/FSDD/recordings'
]

def get_train_dataloader():
    trainsets = [VoiceCommandDataset(d) for d in TRAIN_SET_FOLDER]
    trainset = ConcatDataset(trainsets)

    loader = DataLoader(
                trainset, 
                batch_size=10, 
                shuffle=True
             )
    return loader


# -----------------------------------------------

if __name__ == '__main__':
    import pdb
    import numpy as np
    import librosa
    import matplotlib.pyplot as plt

    loader = get_train_dataloader()

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

        mel_spec = mel_specs[0].permute(1, 0).numpy()
        spec = np.abs(specs[0].permute(1, 0).numpy()) ** 2
        plot_feature2(mel_spec)
        plot_feature2(spec)

        pdb.set_trace()