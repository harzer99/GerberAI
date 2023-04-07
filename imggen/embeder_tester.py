from pathlib import Path
import more_itertools
import sys
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
import matplotlib.pyplot as plt


samples = 50
songs = 10

input_path = sys.argv[1] if len(sys.argv) > 1 else 'C:\\GerberAI'
images, audio_embeddings = torch.load(input_path + '\\dataset_debug.pt').tensors
audio_embeddings = audio_embeddings.cpu()
embedding = audio_embeddings.numpy()
total = len(embedding[:,0])

assert samples*songs == total
stabws_corr = np.zeros(songs)
i = 0
while i < 10:
    deltas = np.zeros(samples-1)
    j = 0
    while j < samples-1:
        deltas[i] =np.linalg.norm(embedding[j + songs*i]-embedding[j + songs*i+1])
        j+=1
    stabws_corr[i] = np.std(deltas)
    i+=1

stabws_rdm = np.zeros(songs)

i = 0
while i < 10:
    deltas = np.zeros(samples-1)
    j = 0
    while j < samples-1:
        rand1 = int(np.random.rand()*total)
        rand2 = int(np.random.rand()*total)
        deltas[i] =np.linalg.norm(embedding[rand1]-embedding[rand2])
        j+=1
    stabws_rdm[i] = np.std(deltas)
    i+=1


plt.plot(stabws_corr, label = 'correlated samples (50 per song)')
plt.plot(stabws_rdm, label = 'random samples (50)')
plt.legend()
plt.xlabel('song number')
plt.ylabel('standard deviation')
plt.show()
print(embedding)