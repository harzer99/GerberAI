import collections
import sys
import time
from imggen.models import MyAudioEmbedder, Generator
import imggen.models as models

sys.modules['models'] = models

import torch
import torchvision.transforms
import torchvision.utils

import numpy as np
import pyaudio
from torch.autograd import Variable

import tkinter as tk
from PIL import Image, ImageTk

GAN_LATENT_DIM = 1000

audio = pyaudio.PyAudio()
for ii in range(audio.get_device_count()):
    print(ii, audio.get_device_info_by_index(ii).get('name'))

SAMPLE_RATE = 48000
CHUNK = 512

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

embedder = MyAudioEmbedder()
embedder.eval()
generator = torch.load(sys.argv[1] if len(sys.argv) > 1 else 'C:\\GerberAI\\generator.pt',
                       map_location=torch.device('cpu'))
if cuda:
    embedder.cuda()
    generator.cuda()

audio_deque = collections.deque(maxlen=int(SAMPLE_RATE / CHUNK * 2.1))


def recorder_callback(in_data, frame_count, time_info, status_flags):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_deque.append(audio_data)
    return None, pyaudio.paContinue


stream = audio.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input_device_index=6,
                    input=True,
                    stream_callback=recorder_callback,
                    frames_per_buffer=CHUNK)

root = tk.Tk()
images = []
labels = []
tk_image = None
tk_label = None
placeholder = ImageTk.PhotoImage(Image.new('RGB', (1024, 576), (0, 0, 255)))
images.append(placeholder)
label = tk.Label(root, image=placeholder, borderwidth=0)
label.grid(row=0, column=0)
labels.append(label)


def update_images():
    start_t = time.monotonic()
    t = start_t
    print(f'peek buffer; +{0:.2f}ms')
    with torch.no_grad():
        track = FloatTensor(torch.tensor(np.concatenate(audio_deque)))
        track_emb = embedder(track.unsqueeze(-1).unsqueeze(0))
        print(f'embedding done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = start_t

        batch_size = 15
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, GAN_LATENT_DIM))))
        gen_imgs = generator(z, track_emb.expand(batch_size, -1)).cpu()
        print(f'{batch_size} images done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = start_t

        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, padding=0)
        out_img = torchvision.transforms.ToPILImage()(grid)
        print(f'post-processing done; +{(time.monotonic() - t) * 1000:.2f}ms')

        images[0] = ImageTk.PhotoImage(out_img)
        labels[0].config(image=images[0])

    delta = 5 + start_t - time.monotonic()
    print(f'waiting {delta * 1000:.2f}ms')
    root.after(int(delta), update_images)


root.after(5, update_images)
root.mainloop()

# snippets = []
# time.sleep(5)
# for i in range(10):
#     time.sleep(5)
#     print('peek buffer', i)
#     track = np.concatenate(audio_deque)
#     snippets.append(track)
#
# time.sleep(5)
# print('playback')
#
#
# stream_out = audio.open(format=pyaudio.paFloat32,
#                     channels=1,
#                     rate=SAMPLE_RATE,
#                     output_device_index=6,
#                     output=True,
#                     frames_per_buffer=CHUNK)
# for track in snippets:
#     for i in range(0, len(track), CHUNK):
#         stream_out.write(track[i:i+CHUNK].tobytes())
#
# stream_out.close()
