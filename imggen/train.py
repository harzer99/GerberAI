import glob
import sys
from pathlib import Path
import more_itertools
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import resampy as resampy
import soundfile
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import save_image

from .models import TARGET_SR, IMG_SHAPE, TRACK_EMB_DIM, Generator, Discriminator, MyAudioEmbedder

GAN_LATENT_DIM = 1000


def sample_image(generator, train_dataset, test_dataset, n, filename, cuda):
    generator.eval()

    def generate_images(track_embs):
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        if cuda:
            track_embs = track_embs.cuda()
        with torch.no_grad():
            generator.eval()
            z = Variable(FloatTensor(np.random.normal(0, 1, (n, GAN_LATENT_DIM))))
            gen_imgs = generator(z, track_embs)
        return gen_imgs.cpu()

    output_imgs = []

    # sample from train dataset
    real_imgs, track_embs = next(iter(DataLoader(train_dataset, batch_size=n, shuffle=True)))
    gen_imgs = generate_images(track_embs)
    for real_img, gen_img in zip(real_imgs, gen_imgs):
        output_imgs.append(real_img.cpu())
        output_imgs.append(gen_img)

    # repeat from test dataset
    real_imgs, track_embs = next(iter(DataLoader(test_dataset, batch_size=n, shuffle=False)))
    gen_imgs = generate_images(track_embs)
    for real_img, gen_img in zip(real_imgs, gen_imgs):
        output_imgs.append(real_img.cpu())
        output_imgs.append(gen_img)

    save_image(torch.stack(output_imgs).cpu().data, filename, nrow=2, normalize=True, value_range=(0, 255))


def plot_loss(epoch_loss_gen, epoch_loss_dis_real, epoch_loss_dis_fake, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.mean(np.array(epoch_loss_gen), axis=1), label='gen loss')
    ax.plot(np.mean(np.array(epoch_loss_dis_real), axis=1), label='dis loss fake')
    ax.plot(np.mean(np.array(epoch_loss_dis_fake), axis=1), label='dis loss real')
    ax.legend()
    ax.set_yscale('log')

    fig.savefig(filename)
    plt.close('all')

def train(dataset_path, output_path , checkpoint_interval = 20, plot_intervall = 20, image_intervall = 20, n_epochs = 100):
    #set these paths
    #######################################################################################
    #dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'C:\\GerberAI\\dataset.pt'
    #output_path = sys.argv[2] if len(sys.argv) > 1 else 'C:\\GerberAI\\imggen_output'
    #checkpoint_interval = 20
    #plot_intervall = 20
    #image_intervall = 10
    #n_epochs = 100
    #######################################################################################


    checkpoint_path = os.path.join(output_path, 'checkpoints')
    image_path = os.path.join(output_path, 'images')
    plot_path = os.path.join(output_path, 'plots')

    os.makedirs(output_path, exist_ok = True )
    os.makedirs(checkpoint_path, exist_ok = True)
    os.makedirs(image_path, exist_ok = True)
    os.makedirs(plot_path, exist_ok = True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    adversarial_loss = torch.nn.BCEWithLogitsLoss()

    # Initialize generator and discriminator
    generator = Generator(GAN_LATENT_DIM)
    discriminator = Discriminator()

    #generator = torch.load('G:\\ProjectEuler\\techno_scraper\\GerberAI\\imggen_output\\gen.torch')
    #discriminator = torch.load( 'G:\\ProjectEuler\\techno_scraper\\GerberAI\\imggen_output\\dis.torch')

    # Optimizers
    optimizer_G = torch.optim.Adam(list(generator.parameters()), lr=0.0002)
    optimizer_D = torch.optim.Adam(list(discriminator.parameters()), lr=0.0002)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    embedding = MyAudioEmbedder()

    dataset = torch.load(Path(dataset_path))
    test_size = 4
    train_dataset, test_dataset = random_split(dataset, [len(dataset)-test_size, test_size], torch.Generator().manual_seed(666))

    # mean/std for sampling in the embedding space
    _, music_emb = dataset.tensors
    emb_std, emb_mean = torch.std_mean(music_emb.cpu(), axis=0)


    dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    epoch_loss_gen, epoch_loss_dis_real, epoch_loss_dis_fake = [], [], []

    for epoch in tqdm(range(n_epochs), desc='epochs', unit='epoch', leave=True):
        epoch_loss_gen.append([])
        epoch_loss_dis_real.append([])
        epoch_loss_dis_fake.append([])
        generator.train()
        discriminator.train()

        for i, (imgs, track_embs) in tqdm(enumerate(dataloader), desc='batch', total=len(dataloader), leave=False, unit='batch'):

            batch_size = imgs.shape[0]

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            track_embs = Variable(track_embs.type(FloatTensor))

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # -----------------
            #  Train Generator + Embedding
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, GAN_LATENT_DIM))))
            # random_emb = Variable(FloatTensor(np.random.normal(emb_mean, emb_std, (batch_size, TRACK_EMB_DIM))))
            random_emb = track_embs

            # Generate a batch of images
            gen_imgs = generator(z, random_emb)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, random_emb)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, track_embs)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), random_emb)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # tqdm.write(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            # )
            epoch_loss_gen[-1].append(g_loss.item())
            epoch_loss_dis_fake[-1].append(d_fake_loss.item())
            epoch_loss_dis_real[-1].append(d_real_loss.item())

        # saving files every given interval
        if epoch > 0 and epoch % checkpoint_interval == 0:
            torch.save(generator, os.path.join(checkpoint_path, f'generator_{epoch:04d}.pt'))
            torch.save(discriminator, os.path.join(checkpoint_path, f'discriminator_{epoch:04d}.pt'))
            checkpoint = epoch - checkpoint_interval                                                    #checkpoint to load is the next to last interval in case a mode lock is detected
        if epoch > 0 and epoch % plot_intervall == 0:
            plot_loss(epoch_loss_gen, epoch_loss_dis_fake, epoch_loss_dis_real, Path(plot_path) / Path(f'loss{epoch:03d}.png'))
            plot_loss(epoch_loss_gen, epoch_loss_dis_fake, epoch_loss_dis_real, Path(plot_path) / Path(f'loss_current.png'))

        if epoch > 0 and epoch % image_intervall == 0:
            sample_image(generator, train_dataset, test_dataset, 4, Path(image_path) / Path(f'epoch{epoch:03d}.png'), cuda)

        #checking if model is mode locked
        if epoch > 2* checkpoint_interval:
            rolling_fake = np.mean(np.array(epoch_loss_dis_fake)[epoch-5:epoch])
            rolling_real = np.mean(np.array(epoch_loss_dis_real)[epoch-5:epoch])
            if rolling_fake < 10**-8 or rolling_real < 10**-8:
                generator = torch.load(os.path.join(checkpoint_path, f'generator_{checkpoint:04d}.pt'))
                discriminator = torch.load(os.path.join(checkpoint_path, f'discriminator_{checkpoint:04d}.pt'))
                print(f'model has been reverted loaded {checkpoint:04d}')

if __name__ == '__main__':
    train("I:\\GerberAI3\\dataset.pt", 'I:\\GerberAI3\\imggen_output' )
