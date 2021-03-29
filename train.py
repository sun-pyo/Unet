import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from unet import Discriminator

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(G_net,
              D_net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    
    # if net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer
    beta1 = 0.5
    beta2 = 0.999
    G_optimizer = optim.Adam(G_net.parameters(), lr=lr, betas=(beta1, beta2))
    D_optimizer = optim.Adam(D_net.parameters(), lr=lr, betas=(beta1, beta2))

    # loss
    BCE_loss = nn.BCELoss().cuda()
    L1_loss = nn.L1Loss().cuda()

    G_net.train()
    D_net.train()

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []

    for epoch in range(epochs):
        #epoch_loss = 0
        D_losses = []
        G_losses = []
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == G_net.n_channels, \
                    f'Network has been defined with {G_net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if G_net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                D_result = D_net(imgs, true_masks).squeeze()
                D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))

                G_result  = G_net(imgs)
                D_result = D_net(imgs, G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))

                D_train_loss = (D_real_loss + D_fake_loss) * 0.5
                D_train_loss.backward()
                D_optimizer.step()

                train_hist['D_losses'].append(D_train_loss.data)
                D_losses.append(D_train_loss.data)

                #loss = criterion(masks_pred, true_masks)
                #epoch_loss += loss.item()
                #writer.add_scalar('Loss/train', loss.item(), global_step)


                G_net.zero_grad()
                G_result = G_net(imgs)
                D_result = D_net(imgs, G_result).squeeze()

                G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + 100 * L1_loss(G_result, true_masks)
                G_train_loss.backward()
                G_optimizer.step()

                train_hist['G_losses'].append(G_train_loss.data)
                G_losses.append(G_train_loss.data)

                pbar.set_postfix(**{'loss (batch)': G_train_loss.data.item()})
                pbar.update(imgs.shape[0])

                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in G_net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(G_net, val_loader, device)
                    #scheduler.step(val_score)
                    #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    writer.add_images('images', imgs, global_step)
        print('[%d/%d] - loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), epochs, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(G_net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            torch.save(D_net.state_dict(),
                       dir_checkpoint + f'D_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    ndf = 64
    G_net = UNet(n_channels=3, n_classes=1, bilinear=True)
    D_net = Discriminator(ndf)
    logging.info(f'Network:\n'
                 f'\t{G_net.n_channels} input channels\n'
                 f'\t{G_net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if G_net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        G_net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    G_net.to(device=device)
    D_net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(G_net=G_net,
                  D_net=D_net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(G_net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
