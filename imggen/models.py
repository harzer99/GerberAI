import numpy as np
import torch
import torch.nn as nn
import torchopenl3
import torchopenl3.utils

TARGET_SR = 48000
TRACK_EMB_DIM = 6144
IMG_SHAPE = (3, 48, 64)


class MyAudioEmbedder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l3model = torchopenl3.core.load_audio_embedding_model('mel256', 'music', 6144)  # magic numbers

    def forward(self, audio_tracks):
        # audio_tracks.shape == (num_tracks, samples, channels)
        audio = audio_tracks.mean(axis=2)
        num_tracks = audio_tracks.shape[0]
        assert audio_tracks.shape[1] // TARGET_SR == 4
        frames = torchopenl3.utils.preprocess_audio_batch(audio, TARGET_SR, center=False, hop_size=1, sampler=None).to(
            torch.float32
        )

        # embedded = self.l3model(frames).reshape(num_tracks, -1)
        embedded = self.l3model(frames).reshape(num_tracks, 5, -1).mean(axis=1)
        assert embedded.shape[-1] == TRACK_EMB_DIM
        return embedded


class Generator(nn.Module):
    def __init__(self, gan_latent_dim):
        super(Generator, self).__init__()
        ngf = 64
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(gan_latent_dim + TRACK_EMB_DIM, ngf * 8, (3,4), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, track_embeddings):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((track_embeddings, noise), -1).unsqueeze(-1).unsqueeze(-1)
        img = self.model(gen_input)

        img = img * 128 + 128  # rescale to 0-255
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ndf = 8
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(3, ndf, 4, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classify = nn.Sequential(
            nn.Linear(TRACK_EMB_DIM + 2240, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, track_embeddings):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((img.view(img.size(0), -1), track_embeddings), -1)
        # validity = self.model(d_in)
        # return validity
        img_conv = self.conv(img)
        d_in = torch.cat((img_conv.view(img.size(0), -1), track_embeddings), -1)
        validity = self.classify(d_in)
        return validity

