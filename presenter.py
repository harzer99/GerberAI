import collections
import sys
import time
from imggen.models import MyAudioEmbedder, Generator
import imggen.models as models
#from beat_predictor import Beat_detector
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


# images = []
# labels = []
# placeholder = ImageTk.PhotoImage(Image.new('RGB', (64, 64), (0, 0, 255)))
# images.append(placeholder)
# tk_canvas = tk.Canvas(root, bg="black", borderwidth=0)
# tk_canvas.pack(fill=tk.BOTH, expand=tk.YES)
# tk_canvas.create_image(0, 0, image=placeholder, anchor=tk.NW)
# labels.append(tk_canvas)


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
sample_time = 1
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
generator = torch.load('/home/leonardmuller/GerberAI/generator_0480.pt',
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
                    input_device_index = 2, #audio.get_default_output_device_info()['index'],
                    input=True,
                    stream_callback=recorder_callback,
                    frames_per_buffer=CHUNK)

#
# def scale_img(img, scale_factor):
#     rows, cols = img.shape[1:]
#     scaled_rows, scaled_cols = int(rows * scale_factor), int(cols * scale_factor)
#
#     # highpass = np.sqrt((np.arange(rows)[:, np.newaxis] - rows//2)**2 + (np.arange(cols)[np.newaxis, :] - cols//2)**2) > 50
#     upscaled_img = np.zeros((scaled_rows, scaled_cols, 3), dtype=np.uint8)
#
#     for chan in range(0, 3):
#         channel = img[chan]
#         f = np.fft.fft2(channel)
#         fshift = np.fft.fftshift(f)
#         # fshift = np.where(highpass, fshift, 0)
#         padded_spectrum = np.pad(fshift, (((scaled_rows - rows) // 2, (scaled_rows - rows) // 2), ((scaled_cols - cols) // 2, (scaled_cols - cols) // 2)), mode='constant')
#         padded_spectrum_shift = np.fft.ifftshift(padded_spectrum)
#         upscaled_chan = np.abs(np.fft.ifft2(padded_spectrum_shift))
#         upscaled_chan = channel.min() + (upscaled_chan - upscaled_chan.min()) / (upscaled_chan.max() - upscaled_chan.min()) * (channel.max() - channel.min())
#         upscaled_img[:,:,chan] = upscaled_chan
#
#     return upscaled_img


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
        z = FloatTensor(np.random.normal(0, 2, (batch_size, GAN_LATENT_DIM)))
        gen_imgs = generator(z, track_emb.expand(batch_size, -1)).cpu().to(torch.uint8)
        print(f'{batch_size} images done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, padding=0)
        # out_img = scale_img(grid.numpy(), 5)
        out_img = torchvision.transforms.ToPILImage()(grid)
        print(f'post-processing done; +{(time.monotonic() - t) * 1000:.2f}ms')
        t = time.monotonic()

        app.change_image(out_img)

        # images[0] = ImageTk.PhotoImage(out_img)
        # labels[0].config(image=images[0])

    delta = 0.1 + start_t - time.monotonic()
    print(f'waiting {delta * 1000:.2f}ms')
    root.after(int(delta*1000), update_images)


root.after(1000, update_images)
root.mainloop()
