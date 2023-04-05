import sys
from pathlib import Path

from models import MyAudioEmbedder
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm
import numpy as np
import resampy as resampy
import soundfile
import torch
import more_itertools
from torch.utils.data import TensorDataset

from models import TARGET_SR, IMG_SHAPE

input_path = sys.argv[1]
output_path = sys.argv[2]

cuda = True if torch.cuda.is_available() else False

embedding = MyAudioEmbedder()
embedding.eval()
if cuda:
    embedding.cuda()

transform = transforms.Compose([transforms.Resize(list(IMG_SHAPE)[1:])])

images, tracks, emb = [], [], []
files = list(Path(input_path).glob('*.wav'))
for wav in tqdm(files, desc='image processing'):
    audio, sr = soundfile.read(wav)
    padding = sr * 5 - audio.shape[0]
    audio = np.pad(audio, ((0, padding), (0, 0)))
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

dataset = TensorDataset(torch.stack(images), torch.stack(emb))
torch.save(dataset, Path(output_path))

