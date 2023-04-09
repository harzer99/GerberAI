import collections
import sys
import time
from imggen.models import MyAudioEmbedder, Generator
import imggen.models as models
import resampy

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

audio_deque = collections.deque(maxlen=int(44100 / CHUNK * 3.8))

def recorder_callback(in_data, frame_count, time_info, status_flags):
    audio_data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, 2)
    audio_deque.append(audio_data)
    return None, pyaudio.paContinue


stream = audio.open(format=pyaudio.paFloat32,
                    channels=2,
                    rate=44100,
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


def scale_img(img, scale_factor):
    rows, cols = img.shape[1:]
    scaled_rows, scaled_cols = int(rows * scale_factor), int(cols * scale_factor)

    upscaled_img = np.zeros((scaled_rows, scaled_cols, 3), dtype=np.uint8)

    for chan in range(0, 3):
        channel = img[chan]
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        padded_spectrum = np.pad(fshift, (((scaled_rows - rows) // 2, (scaled_rows - rows) // 2), ((scaled_cols - cols) // 2, (scaled_cols - cols) // 2)), mode='constant')
        padded_spectrum_shift = np.fft.ifftshift(padded_spectrum)
        upscaled_chan = np.real(np.fft.ifft2(padded_spectrum_shift))
        upscaled_chan = channel.min() + (upscaled_chan - upscaled_chan.min()) / (upscaled_chan.max() - upscaled_chan.min()) * (channel.max() - channel.min())
        upscaled_img[:,:,chan] = upscaled_chan

    return upscaled_img


def update_images():
    start_t = time.monotonic()
    t = start_t
    print(f'peek buffer; +{0:.2f}ms')
    with torch.no_grad():
        embedder.eval()
        generator.eval()
        audio = np.concatenate(audio_deque)
        track = torch.tensor(resampy.resample(
            audio,
            sr_orig=44100,
            sr_new=48000,
            filter='kaiser_fast'
        ))
        if cuda:
            track = track.cuda()
        print(f'audio preprocessing; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        track_emb = embedder(torch.stack([track]))
        print(f'embedding done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        batch_size = 15
        z = FloatTensor(np.random.normal(0, 10, (batch_size, GAN_LATENT_DIM)))
        gen_imgs = generator(z, track_emb.expand(batch_size, -1)).cpu().to(torch.uint8)
        print(f'{batch_size} images done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, padding=0)
        out_img = scale_img(grid.numpy(), 5)
        out_img = Image.fromarray(out_img)
        print(f'post-processing done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        images[0] = ImageTk.PhotoImage(out_img)
        labels[0].config(image=images[0])

    delta = 2 + start_t - time.monotonic()
    print(f'waiting {delta * 1000:.2f}ms')
    root.after(int(delta*1000), update_images)


root.after(1000, update_images)
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
