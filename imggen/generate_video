import glob
import sys
from pathlib import Path
import more_itertools
import matplotlib
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
import subprocess

from models import TARGET_SR, IMG_SHAPE, TRACK_EMB_DIM, Generator, Discriminator, MyAudioEmbedder
matplotlib.use("Agg")

GAN_LATENT_DIM = 1000

#set these paths
#######################################################################################
song_path = sys.argv[1] if len(sys.argv) > 1 else "E:\\musik\\mix der Woche Sammlung\\Harold Alexander\\Sunshine Man\\04 Mama Soul.mp3" #"I:\\GerberAI\\downloaded_audios\\4OQXjGPs7qI.webm"
output_path = sys.argv[2] if len(sys.argv) > 1 else 'I:\\GerberAI\\output_video'
img_cache = output_path + '\\cache'
converted_path = output_path + '\\sample.wav'
generator_path = sys.argv[2] if len(sys.argv) > 1 else 'I:\\GerberAI\\imggen_output\\checkpoints\\generator_0600.pt'
#discriminator_path = sys.argv[2] if len(sys.argv) > 1 else 'I:\\GerberAI\\imggen_output\\checkpoints\\discriminator_0200.pt'
load_existing = True

#######################################################################################

cuda = True if torch.cuda.is_available() else False
embedding = MyAudioEmbedder()
embedding.eval()
if cuda:
    embedding.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(generator, track_embs, cuda):
    generator.eval()
    def generate_images(track_embs):
        if cuda:
            track_embs = track_embs.cuda()
        with torch.no_grad():
            generator.eval()
            z = Variable(FloatTensor(np.random.normal(0, 1, (1, GAN_LATENT_DIM))))
            gen_imgs = generator(z, track_embs)
        return gen_imgs.cpu()

    # sample from train dataset
    gen_img = generate_images(track_embs)
    return gen_img
    #save_image(torch.stack(output_imgs).cpu().data, filename, nrow=2, normalize=True, value_range=(0, 255))

def embedd(audio, sr):
    padding = sr * 5 - audio.shape[0]
    audio = np.pad(audio, ((0, padding), (0, 0)))
    # resample to TARGET_SR = 48000
    
    track = torch.tensor(resampy.resample(
        audio,
        sr_orig=sr,
        sr_new=TARGET_SR,
        filter="kaiser_best",
    )).to(torch.float32)
    #tracks.append(track)
    track_batch = torch.stack([track])
    if cuda:
        track_batch = track_batch.cuda()
        return embedding(track_batch)
    else:
        return embedding(track_batch)

def generate_sampples(song, sr, window, fps):
    frame_interval = int(1/fps*sr)
    window = window*sr
    length = len(song)
    seek = int(window)
    samples = np.zeros([int(length/frame_interval), window, 2])
    i = 0
    while seek < length:
        samples[i] = song[seek-window:seek]
        seek += frame_interval
        i += 1

    return samples 

def cache_images (song):
    samples = generate_sampples(song, sr, 5, fps)
    embeddings = np.zeros([len(samples[:,0]), 512])
    images = []
    j = 0
    for j in tqdm (range(len(samples[:,0]))):
        emb = embedd(samples[j], sr)
        image = sample_image(generator, emb, cuda)
        save_image(image.cpu().data, img_cache + f'\\{j:03d}.png', nrow=1, normalize=True, value_range=(0, 255))
        j += 1

generator = torch.load(generator_path) 
cmd = ['ffmpeg', '-hide_banner', '-nostats', '-y','-i', song_path, '-ar', '44100',  converted_path]
subprocess.run(cmd)
samplewindow = 5
fps = 10
song, sr = soundfile.read(converted_path)
cache_images(song)

video_command = [
    'ffmpeg', '-y',
    '-framerate', f'{fps}',
    '-f', 'image2',
    '-i', img_cache + '\\%03d.png',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    output_path+'\\output.mp4'
]

audio_command = [
    'ffmpeg', '-y',
    '-i', output_path+'\\output.mp4',
    '-i', converted_path,
    '-ss', '5',
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-b:a', '256k',
    output_path+'\\output_with_audio.mp4'
]
subprocess.call(video_command)
subprocess.call(audio_command)
print(max(song[:,0]))





