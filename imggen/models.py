import numpy as np
import torch
import torch.nn as nn
import torchopenl3
import torchopenl3.utils

TARGET_SR = 48000
TRACK_EMB_DIM = 5 * 6144
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

        embedded = self.l3model(frames).reshape(num_tracks, -1)
        assert embedded.shape[-1] == TRACK_EMB_DIM
        return embedded


class Generator(nn.Module):
    def __init__(self, gan_latent_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(gan_latent_dim + TRACK_EMB_DIM, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(IMG_SHAPE))),
            nn.Tanh(),
        )

    def forward(self, noise, track_embeddings):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((track_embeddings, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *IMG_SHAPE)

        img = img * 128 + 128  # rescale to 0-255
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(TRACK_EMB_DIM + int(np.prod(IMG_SHAPE)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, track_embeddings):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), track_embeddings), -1)
        validity = self.model(d_in)
        return validity

