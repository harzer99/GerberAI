import glob
import sys
from pathlib import Path
import more_itertools

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

from models import TARGET_SR, IMG_SHAPE, TRACK_EMB_DIM, Generator, Discriminator, MyAudioEmbedder

GAN_LATENT_DIM = 1000


def load_dataset(embedding, cuda):
    embedding.eval()
    if cuda:
        embedding.cuda()
    transform = transforms.Compose([transforms.Resize(list(IMG_SHAPE)[1:])])
    images, tracks, emb = [], [], []
    files = list((Path('imggen') / Path('data')).glob('*.wav'))
    for wav in tqdm(files, desc='image processing'):
        audio, sr = soundfile.read(wav)
        # resample to TARGET_SR = 48000
        track = torch.tensor(resampy.resample(
                audio,
                sr_orig=sr,
                sr_new=TARGET_SR,
                filter="kaiser_best",
            )).to(torch.float32)

        img = transform(read_image(str(wav).replace('.wav', '.png')))

        images.append(img)
        tracks.append(track)

    with torch.no_grad():
        with tqdm(total=len(tracks), desc='music embedding') as pbar:
            for track_batch in more_itertools.chunked(tracks, n=4 if cuda else 1):
                track_batch = torch.stack(track_batch)
                if cuda:
                    track_batch = track_batch.cuda()
                    emb.extend(embedding(track_batch))
                else:
                    emb.extend(embedding(track_batch))

                pbar.update(len(track_batch))
                del track_batch

    if cuda:
        return TensorDataset(torch.stack(images).cuda(), torch.stack(emb).cuda())
    else:
        return TensorDataset(torch.stack(images), torch.stack(emb))


def sample_image(generator, train_dataset, test_dataset, n, filename, cuda):
    generator.eval()

    def generate_images(track_embs):
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
        output_imgs.append(real_img)
        output_imgs.append(gen_img)

    # repeat from test dataset
    real_imgs, track_embs = next(iter(DataLoader(test_dataset, batch_size=n, shuffle=False)))
    gen_imgs = generate_images(track_embs)
    for real_img, gen_img in zip(real_imgs, gen_imgs):
        output_imgs.append(real_img)
        output_imgs.append(gen_img)

    save_image(torch.stack(output_imgs).cpu().data, filename, nrow=2, normalize=True, value_range=(0, 255))


cuda = True if torch.cuda.is_available() else False

embedding = MyAudioEmbedder()

dataset = load_dataset(embedding, cuda)
test_size = 4
train_dataset, test_dataset = random_split(dataset, [len(dataset)-test_size, test_size])

# mean/std for sampling in the embedding space
_, music_emb = dataset.tensors
emb_std, emb_mean = torch.std_mean(music_emb.cpu(), axis=0)

# Loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
generator = Generator(GAN_LATENT_DIM)
discriminator = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(list(generator.parameters()), lr=0.0002)
optimizer_D = torch.optim.Adam(list(discriminator.parameters()), lr=0.0002)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

n_epochs = 200
for epoch in tqdm(range(n_epochs)):
    generator.train()
    discriminator.train()
    for i, (imgs, track_embs) in enumerate(dataloader):

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

    if epoch > 0 and epoch % 10 == 0:
        sample_image(generator, train_dataset, test_dataset, 4, Path('imggen')/Path('output') / Path(f'epoch{epoch:03d}.png'), cuda)
