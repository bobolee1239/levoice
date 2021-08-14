# ----------------------------------
# File: train_LeVoice.py
# ----------------------------------
import os
import sys
import torch
import numpy       as np

from tensorboardX import SummaryWriter
from tqdm         import tqdm

if '..' not in sys.path:
    sys.path.append('..')

from dataset.Dataloader import get_train_dataloader
from model.LeVoice      import LeVoice

import config
# ---------------------------------------------------------------
if torch.cuda.is_available():
    print('[INFO] Training with GPU')
    device = torch.device('cuda')
else:
    print('[WARN] Training with CPU')
    device = torch.device('cpu')

nfreq = config.NMEL

model = LeVoice(nfreq).float()
model.to(device)
optim = torch.optim.RMSprop(
            model.parameters(),
            weight_decay=1e-4
        )
critera = torch.nn.CrossEntropyLoss()
# ---------------------------------------------------------------

def load_model(net, path):
    net.load_state_dict(torch.load(path))
    
def run_batch(batch):
    sigs, spectrums, mel_spectras, labels = batch

    mel_spectras = mel_spectras.to(device)
    labels       = labels.to(device)

    preds = model(mel_spectras)
    loss = critera(preds.reshape(-1, config.N_CLASS), labels.reshape(-1))

    return loss, preds

def train(epoch, save_dir='.'):
    nstep = 0
    running_loss = 0.0
    nlog  = 200
    nsave = 2000 
    batch_size = 64

    model.train()
    train_loader = get_train_dataloader(batch_size=batch_size, nworker=8)
    for n in range(epoch):
        loader = tqdm(train_loader)

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

            to_save = (nstep - 1) % nsave == 0
            if to_save:
                savefile = os.path.join(save_dir, f'LeVoice-{nstep}.pth')
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir, exist_ok=False)
                torch.save(model.state_dict(), savefile)
    return nstep


def main(args):
    nepoch     = args.epoch
    model_path = args.load
    save_dir   = args.save_dir

    nstep = -1
    if model_path:
        load_model(model, model_path)
    try:
        nstep = train(nepoch, 
                      save_dir=save_dir)
    except KeyboardInterrupt as err:
        print(err)
        pass

    savefile = os.path.join(save_dir, f'LeVoice-{nstep}.pth')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=False)
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