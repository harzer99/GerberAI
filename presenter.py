import collections
import sys
import time

import librosa as librosa

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


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")
        self.master.geometry("800x600")

        # Load image
        self.img = Image.new('RGB', (64, 64), (0, 0, 255))
        # self.img = Image.open("example.jpg")
        self.photo = ImageTk.PhotoImage(self.img)

        # Create canvas to display image
        self.canvas = tk.Canvas(self.master, bg="black", borderwidth=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

        # Display image
        self.canvas.bind('<Configure>', self.resize_canvas)

    def change_image(self, image):
        self.img = image
        self.photo = ImageTk.PhotoImage(self.img)
        self.resize_image()

    def resize_canvas(self, event):
        self.canvas.config(width=event.width, height=event.height)
        self.resize_image()

    def resize_image(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        aspect_ratio = self.img.width / self.img.height
        if canvas_width / canvas_height < aspect_ratio:
            size = (canvas_width, int(canvas_width / aspect_ratio))
        else:
            size = (int(canvas_height * aspect_ratio), canvas_height)
        resized_img = self.img.resize(size, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(resized_img)
        x = (canvas_width - resized_img.width) // 2
        y = (canvas_height - resized_img.height) // 2

        self.canvas.delete('all')
        self.canvas.create_image(x, y, image=self.photo, anchor=tk.NW)


root = tk.Tk()
app = App(root)

GAN_LATENT_DIM = 1000
CHUNK = 512
sample_time = 20
embedding_time = 5
calibration_shift = 0.02
audio = pyaudio.PyAudio()
for ii in range(audio.get_device_count()):
    print(ii, audio.get_device_info_by_index(ii).get('name'))

device_info = audio.get_device_info_by_index(2)
print(device_info)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

embedder = MyAudioEmbedder()
embedder.eval()
generator = torch.load('/home/leonardmuller/GerberAI/generators_leo/generator_0480.pt',
                       map_location=torch.device('cpu'))
if cuda:
    embedder.cuda()
    generator.cuda()

audio_deque = collections.deque(maxlen=int(44100 / CHUNK * sample_time))

def recorder_callback(in_data, frame_count, time_info, status_flags):
    audio_data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, 2)
    audio_deque.append(audio_data)
    return None, pyaudio.paContinue


stream = audio.open(format=pyaudio.paFloat32,
                    channels=2,
                    rate=44100,
                    input_device_index = 3, #audio.get_default_output_device_info()['index'],
                    input=True,
                    stream_callback=recorder_callback,
                    frames_per_buffer=CHUNK)


def update_images():
    start_t = time.monotonic()
    t = start_t
    print(f'peek buffer; +{0:.2f}ms')
    with torch.no_grad():
        embedder.eval()
        generator.eval()

        audio = np.concatenate(audio_deque)

        tempo, beats = librosa.beat.beat_track(y=audio.mean(axis=1), sr=44100, start_bpm=120)
        if len(beats) == 0:
            beats_time = [0]
            print('warn: no beats found, defaulting to start of buffer')
        else:
            beats_time = librosa.frames_to_time(beats, sr=44100)
        meanbeat = start_t - len(audio) / 44100 + beats_time[-1]

        if tempo != 0:
            tau = 1 / tempo * 60
        else:
            tau = 2
            print('warn: no tempo found, defaulting to tau=2s')
        print(f'beat detection (tempo: {tempo:.2f}bpm; tau: {tau:.2f}); +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        track = torch.tensor(resampy.resample(
            audio[-embedding_time*44100:],
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
        z = FloatTensor(np.random.normal(0, 2, (batch_size, GAN_LATENT_DIM)))
        gen_imgs = generator(z, track_emb.expand(batch_size, -1)).cpu().to(torch.uint8)
        print(f'{batch_size} images done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, padding=0)
        # out_img = scale_img(grid.numpy(), 5)
        out_img = torchvision.transforms.ToPILImage()(grid)
        print(f'post-processing done; +{(time.monotonic() - t) * 1000:.2f}ms')

        next_beat = None
        for i in range(100):
            if meanbeat + i*tau - 0.020-calibration_shift > time.monotonic():
                next_beat = meanbeat + i*tau
                break

        if next_beat is not None:
            beat_delta = next_beat - time.monotonic()
            beat_delta = min(beat_delta, 3)-calibration_shift
            print(f'waiting {beat_delta*1000:.2f}ms to next beat')
            time.sleep(beat_delta)
        else:
            print('warn: no beat found!')

        app.change_image(out_img)

    print(f'waiting {10:.2f}ms to next invocation')
    root.after(10, update_images)


root.after(1000, update_images)
root.mainloop()
