import collections
import sys
import time
from pathlib import Path
import os
import librosa as librosa
sys.path.append(os.getcwd())
sys.path.append("./imggen")
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

GAN_LATENT_DIM = 1000
CHUNK = 512


def recorder_callback(in_data, frame_count, time_info, status_flags):
    audio_data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, 2)
    audio_deque.append(audio_data)
    return None, pyaudio.paContinue

def get_mean_beat (audio, start_t   ):
    #print('launcehd beatanalys')
    tempo, beats = librosa.beat.beat_track(y=audio.mean(axis=1), sr=44100, start_bpm=120)
    #print(tempo, beats)
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

#putting the beat analysis on a different process
def my_thread_func(beat_arg_queue, beat_result_queue):
    while True:
        
        # Wait for new arguments to be added to the queue
        #print(__name__)
        args = beat_arg_queue.get()
        t = time.monotonic()
        #print('got args')
        # Execute the function with the new arguments
        result = get_mean_beat(*args)

        # Add the result to the result queue
        
        beat_result_queue.put(result)
        #print(time.monotonic()-t)

        time.sleep(0)
    return

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

def update_images():
    if __name__ == '__mp_main__':
        return
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
        

        

        track = torch.tensor(resampy.resample(
            audio[-int(embedding_time*44100):],
            sr_orig=44100,
            sr_new=48000,
            filter='kaiser_fast'
        ))
        
        if cuda:
            track = track.cuda()
        t_gen = time.monotonic()
        track_emb = embedder(torch.stack([track]))#+ torch.tensor(np.random.normal(0, chaos, (1, 512))).cuda().float()
        t_gen = time.monotonic()-t_gen
        
        z = FloatTensor(np.random.normal(0, z_height, (batch_size, GAN_LATENT_DIM)))
        gen_imgs = generator(z, track_emb.expand(batch_size, -1)).cpu().to(torch.uint8)

        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, padding=0)
        out_img = torchvision.transforms.ToPILImage()(grid)
        beat_time = time.monotonic()-beat_start
        
        t_get = time.monotonic()
        
        meanbeat, tau = beat_result_queue.get()
        t_get = time.monotonic()-t_get
        bmp = 1/tau*60
        next_beat = None
        for i in range(100):
            if meanbeat + (i)*tau - 0-calibration_shift > time.monotonic():
                next_beat = meanbeat + i*tau
                break

        if next_beat is not None:
            beat_delta = next_beat - time.monotonic()
            beat_delta = min(beat_delta, 1)-calibration_shift
            print(f'waiting {beat_delta*1000:.2f}ms to next beat, runtime {beat_time * 1000:.2f}ms, switching {t_switch*1000:.2f}, getting beats {t_get*1000:.2f}, generating {t_gen*1000:.2f}, nvidia{cuda} generator index {current_generator_index}, bpm{bmp}')
            time.sleep(beat_delta)
        else:
            print('warn: no beat found!')

        app.change_image(out_img)

    #print(f'waiting {10:.2f}ms to next invocation')
    root.after(10, update_images)

def present(generator_directory, audiodevice):

    if __name__ =='__mp_main__':
        return
    global sample_time
    global embedding_time
    global calibration_shift
    global stability
    global chaos
    global z_height
    global batch_size
    global device_index
    global primary_screen_width
    global primary_screen_height 
    global cuda
    global FloatTensor
    global LongTensor
    global stream
    global current_generator_indexg
    global meanbeat
    global current_generator_index
    global tau
    global app
    global root
    global generators
    global embedder
    global beat_arg_queue
    global beat_result_queue


    sample_time = 20            #for the beatanalysis
    embedding_time = 2          #for the audio embedder
    calibration_shift = 0.06    #adjust to compensate for output latency, so that each image is pushed on a beat
    stability = 1             #probability to stay with the current generator
    chaos = 0                   #noise added onto the embedding vector that makes variance between frames larger
    z_height = 1 + chaos        #noise that is added for each subimage. This makes them vary within a frame
    batch_size = 15             #number of subimages in each frame
    device_index = 2            #change to your audio device. it should print a list of audio devices available when you run this script

    # Calculate the position of the window on the second screen
    second_screen_x = 1920
    second_screen_y = 0
    root = tk.Tk()
    # Get the width and height of the primary screen
    primary_screen_width = root.winfo_screenwidth()
    primary_screen_height = root.winfo_screenheight()

    # Set the window attributes to fullscreen
    root.attributes("-fullscreen", True)

    # Set the window geometry to the position of the second screen
    root.geometry("1920x1080+{}+{}".format(second_screen_x, second_screen_y))
    app = App(root)


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

    #Change this path to your folder
    generators = []
    for file in sorted(Path(generator_directory).glob('generator_*.pt')):
        generators.append(torch.load(file, map_location=torch.device('cpu')))

    if cuda:
        embedder.cuda()

    global audio_deque
    audio_deque = collections.deque(maxlen=int(44100 / CHUNK * sample_time))

    stream = audio.open(format=pyaudio.paFloat32,
                    channels=2,
                    rate=44100,
                    input_device_index = device_index, #audio.get_default_output_device_info()['index'],
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
    


    root.after(500, update_images)
    root.mainloop()

if __name__ == '__main__':
    present('I:\\GerberAI\imggen_output\checkpoints', 16)