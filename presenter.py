import collections
import sys
import time
from pathlib import Path

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
import multiprocessing



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
# Get the width and height of the primary screen
primary_screen_width = root.winfo_screenwidth()
primary_screen_height = root.winfo_screenheight()

# Set the window attributes to fullscreen
root.attributes("-fullscreen", True)

#canvas = tk.Canvas(root)
#canvas.pack(fill=tk.BOTH, expand=True)

# Calculate the position of the window on the second screen
second_screen_x = 1920
second_screen_y = 0

# Set the window geometry to the position of the second screen
root.geometry("1920x1080+{}+{}".format(second_screen_x, second_screen_y))
app = App(root)

GAN_LATENT_DIM = 1000
CHUNK = 512
sample_time = 20
embedding_time = 2
calibration_shift = 0.12
stability = 0.8
chaos = 0
z_height = 1 + chaos
batch_size = 15

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

generators = []
for file in sorted(Path('/home/leonardmuller/GerberAI/generators/').glob('generator_*.pt')):
    generators.append(torch.load(file, map_location=torch.device('cpu')))

if cuda:
    embedder.cuda()

audio_deque = collections.deque(maxlen=int(44100 / CHUNK * sample_time))

def recorder_callback(in_data, frame_count, time_info, status_flags):
    audio_data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, 2)
    audio_deque.append(audio_data)
    return None, pyaudio.paContinue

def get_mean_beat (audio, start_t   ):
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
    return (meanbeat, tau)

def my_thread_func(beat_arg_queue, beat_result_queue):
    while True:
        
        # Wait for new arguments to be added to the queue

        args = beat_arg_queue.get()
        t = time.monotonic()
        # Execute the function with the new arguments
        result = get_mean_beat(*args)

        # Add the result to the result queue
        if not beat_result_queue.empty():
            old = beat_result_queue.get()
        beat_result_queue.put(result)
        print(time.monotonic()-t)

        time.sleep(0.1)
    return


if __name__ == '__main__':
    stream = audio.open(format=pyaudio.paFloat32,
                    channels=2,
                    rate=44100,
                    input_device_index = 2, #audio.get_default_output_device_info()['index'],
                    input=True,
                    stream_callback=recorder_callback,
                    frames_per_buffer=CHUNK)
    multiprocessing.set_start_method('spawn')
    beat_result_queue = multiprocessing.Queue()
    beat_arg_queue = multiprocessing.Queue()
    
    beat_process= multiprocessing.Process(target = my_thread_func, args=(beat_arg_queue, beat_result_queue,))
    beat_process.start()
    current_generator_index = len(generators) // 2
    meanbeat = time.monotonic()-1
    tau = 1
    def update_images():
        global current_generator_index
        start_t = time.monotonic()
        t = start_t
        #print(f'peek buffer; +{0:.2f}ms')
        with torch.no_grad():
            beat_start = time.monotonic()
            
            audio = np.concatenate(audio_deque)
            beat_arg_queue.put((audio, start_t))
            t_switch = time.monotonic()
            embedder.eval()
            r = np.random.choice([-1, 0, 1], p=[(1-stability)/2, stability, (1-stability)/2])
            if r != 0:
                generators[current_generator_index].cpu()

            current_generator_index = current_generator_index + r
            if current_generator_index < 0:
                current_generator_index = 0
            if current_generator_index >= len(generators):
                current_generator_index = len(generators) - 1

            generator = generators[current_generator_index]
            if cuda:
                generator.cuda()
            generator.eval()

            t_switch = time.monotonic()-t_switch
            

            
            
            #print(f'beat detection (tempo: {tempo:.2f}bpm; tau: {tau:.2f}); +{(time.monotonic() - t) * 1000:.2f}ms')
            

            track = torch.tensor(resampy.resample(
                audio[-int(embedding_time*44100):],
                sr_orig=44100,
                sr_new=48000,
                filter='kaiser_fast'
            ))
            
            if cuda:
                track = track.cuda()
            #print(f'audio preprocessing; +{(time.monotonic() - t) * 1000:.2f}ms')
            t = time.monotonic()
            t_gen = time.monotonic()
            track_emb = embedder(torch.stack([track]))#+ torch.tensor(np.random.normal(0, chaos, (1, 512))).cuda().float()
            #print(f'embedding done; +{(time.monotonic() - t) * 1000:.2f}ms')
            t = time.monotonic()
            t_gen = time.monotonic()-t_gen
            
            z = FloatTensor(np.random.normal(0, z_height, (batch_size, GAN_LATENT_DIM)))
            gen_imgs = generator(z, track_emb.expand(batch_size, -1)).cpu().to(torch.uint8)
            #print(f'{batch_size} images done; +{(time.monotonic() - t) * 1000:.2f}ms')
            t = time.monotonic()
            grid = torchvision.utils.make_grid(gen_imgs, nrow=5, padding=0)
            # out_img = scale_img(grid.numpy(), 5)
            out_img = torchvision.transforms.ToPILImage()(grid)
            #print(f'post-processing done; +{(time.monotonic() - t) * 1000:.2f}ms')
            beat_time = time.monotonic()-beat_start
            
            t_get = time.monotonic()
            
            meanbeat, tau = beat_result_queue.get()
            t_get = time.monotonic()-t_get

            next_beat = None
            for i in range(100):
                if meanbeat + (i)*tau - 0-calibration_shift > time.monotonic():
                    next_beat = meanbeat + i*tau
                    break

            if next_beat is not None:
                beat_delta = next_beat - time.monotonic()
                beat_delta = min(beat_delta, 1)-calibration_shift
                print(f'waiting {beat_delta*1000:.2f}ms to next beat, runtime {beat_time * 1000:.2f}ms, switching {t_switch*1000:.2f}, getting beats {t_get*1000:.2f}, generating {t_gen*1000:.2f}, nvidia{cuda} generator index {current_generator_index}')
                time.sleep(beat_delta)
            else:
                print('warn: no beat found!')

            app.change_image(out_img)

        #print(f'waiting {10:.2f}ms to next invocation')
        root.after(10, update_images)


    root.after(500, update_images)
    root.mainloop()
