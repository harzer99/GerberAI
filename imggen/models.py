import numpy as np
import torch
import torch.nn as nn
import torchopenl3
import torchopenl3.utils

TARGET_SR = 48000
TRACK_EMB_DIM = 512
IMG_SHAPE = (3, 64, 64)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class MyAudioEmbedder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l3model = torchopenl3.core.load_audio_embedding_model('mel256', 'music', TRACK_EMB_DIM)  # magic numbers

    def forward(self, audio_tracks):
        # audio_tracks.shape == (num_tracks, samples, channels)
        audio = audio_tracks.mean(axis=2)
        num_tracks = audio_tracks.shape[0]
        # assert audio_tracks.shape[1] // TARGET_SR == 4
        frames = torchopenl3.utils.preprocess_audio_batch(audio, TARGET_SR, center=False, hop_size=1, sampler=None).to(
            torch.float32
        )

        # embedded = self.l3model(frames).reshape(num_tracks, -1)
        embedded = self.l3model(frames).reshape(num_tracks, -1, TRACK_EMB_DIM).mean(axis=1)
        assert embedded.shape[-1] == TRACK_EMB_DIM
        return embedded


class Generator(nn.Module):
    def __init__(self, gan_latent_dim):
        super(Generator, self).__init__()
        ngf = 32
        self.model = nn.Sequential(
            nn.ConvTranspose2d(gan_latent_dim + TRACK_EMB_DIM, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, noise, track_embeddings):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((track_embeddings, noise), -1).unsqueeze(-1).unsqueeze(-1)
        img = self.model(gen_input)

        img = img * 128 + 128  # rescale to [0 255]
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ndf = 32
        self.image_embedder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(ndf * 16 + TRACK_EMB_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )
        self.apply(weights_init)

    def forward(self, img, track_embeddings):
        img = (img.to(torch.float32) - 128)/128  # rescale to [-1 1]
        img_emb = self.image_embedder(img).reshape(img.shape[0], -1)
        probits = self.predictor(torch.cat((img_emb, track_embeddings), dim=1))
        return probits

