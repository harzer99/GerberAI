import glob
import sys
from pathlib import Path

import numpy as np
import resampy as resampy
import soundfile
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import save_image

from models import TARGET_SR, IMG_SHAPE, TRACK_EMB_DIM, Generator, Discriminator, MyAudioEmbedder

GAN_LATENT_DIM = 1000
TRAIN_EMBEDDING = False

def load_dataset():
    transform = transforms.Compose([transforms.Resize(list(IMG_SHAPE)[1:])])
    images, tracks = [], []
    for index in range(12):
        audio, sr = soundfile.read(f'./data/{index:03d}.wav')
        # resample to TARGET_SR = 48000
        track = torch.tensor(resampy.resample(
                audio,
                sr_orig=sr,
                sr_new=TARGET_SR,
                filter="kaiser_best",
            )).to(torch.float32)

        img = transform(read_image(f'./data/{index:03d}.png'))

        images.append(img)
        tracks.append(track)

    return TensorDataset(torch.stack(images), torch.stack(tracks))


def sample_image(generator, embedding, dataset, n, filename, cuda):
    generator.eval()
    embedding.eval()

    # sample from dataset
    real_imgs, tracks = next(iter(DataLoader(dataset, batch_size=n, shuffle=True)))

    if cuda:
        real_imgs = real_imgs.cuda()
        tracks = tracks.cuda()

    with torch.no_grad():
        track_embs = embedding(tracks)
        z = Variable(FloatTensor(np.random.normal(0, 1, (n, GAN_LATENT_DIM))))
        gen_imgs = generator(z, track_embs)

    output_imgs = []
    for i in range(n):
        output_imgs.append(real_imgs[i])
        output_imgs.append(gen_imgs[i])


    save_image(torch.stack(output_imgs).cpu().data, filename, nrow=2, normalize=True, value_range=(0, 255))



dataset = load_dataset()
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# Loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator and embedding
generator = Generator(GAN_LATENT_DIM)
discriminator = Discriminator()
embedding = MyAudioEmbedder()

# Optimizers
if TRAIN_EMBEDDING:
    optimizer_G = torch.optim.Adam(list(generator.parameters()) + list(embedding.parameters()), lr=0.0002)
else:
    optimizer_G = torch.optim.Adam(list(generator.parameters()), lr=0.0002)
optimizer_D = torch.optim.Adam(list(discriminator.parameters()), lr=0.0002)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if cuda:
    generator.cuda()
    discriminator.cuda()
    embedding.cuda()
    adversarial_loss.cuda()



n_epochs = 200
for epoch in range(n_epochs):
    generator.train()
    discriminator.train()
    if TRAIN_EMBEDDING:
        embedding.train()
    else:
        embedding.eval()
    for i, (imgs, tracks) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        tracks = Variable(tracks.type(FloatTensor))

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator + Embedding
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, GAN_LATENT_DIM))))
        # eigentlich müssten wir hier zufällige Embeddings aus dem Vektorraum ziehen
        if TRAIN_EMBEDDING:
            track_embs = embedding(tracks)
        else:
            with torch.no_grad():
                track_embs = embedding(tracks).detach()

        # Generate a batch of images
        gen_imgs = generator(z, track_embs)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, track_embs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, track_embs.detach())
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), track_embs.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

    sample_image(generator, embedding, dataset, 4, f'output/epoch{epoch:03d}.png', cuda)
