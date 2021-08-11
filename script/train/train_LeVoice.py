# ----------------------------------
# File: train_LeVoice.py
# ----------------------------------
import os
import torch
import numpy       as np

from tqdm import tqdm

from dataset.Dataloader import get_train_dataloader
from model.LeVoice      import LeVoice

# ---------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nfreq = 40

model = LeVoice(nfreq)
optim = torch.optim.RMSProp(
            model.parameters(),
            weight_decay=1e-4
        )
critera = torch.nn.CrossEntropyLoss()
# ---------------------------------------------------------------
def run_batch(batch):
    sigs, spectrums, mel_spectras, labels = batch

    preds = model(mel_spectras)
    loss = criteria(preds, labels)

    return loss, preds

def train(epoch):
    nstep = 0
    running_loss = 0.0
    nlog = 200

    for n in range(epoch):
        loader = tqdm(get_train_dataloader())

        for batch in loader:
            optim.zero_grad()
            loss, preds = run_batch(batch)

            loss.backward()
            optim.step()
            running_loss += loss.item()

            to_log = (nstep - 1) % nlog == 0
            if to_log:    
                loader.set_postfix({
                    'loss': running_loss / float(nlog)
                })
                running_loss = 0.0

        nstep += 1


def main(args):
    nepoch     = args.epoch
    load_model = args.load
    save_dir   = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    train(nepoch)

    savefile = os.path.join(save_dir, 'LeVoice.pth')
    torch.save(model.state_dict(), savefile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--load', 
        type=str,
        default='',
        help='Pretrained model to be loaded'
    )
    parser.add_argument(
        '-e',
        '--epoch', 
        type=int,
        default=100,
        help='# Training Epoch'
    )
    parser.add_argument(
        '--save-dir', 
        type=str,
        default='checkpoint',
        help='Folder to save checkpoints'
    )

    args = parser.parse_args()

    main(args)