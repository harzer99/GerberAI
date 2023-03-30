import numpy as np
from scipy.io.wavfile import write

rate = 22050
stream = np.load('E:\musik\\trainingsmusic\\audio.npy')
beats = np.load('E:\musik\\trainingsmusic\\flags.npy')
beeplength = int(0.1*rate)
samples = np.linspace(0, beeplength, beeplength)
beep = np.sin(2 * np.pi * 440/rate*(2**1.5) * samples)

for beat in beats:
    offset = beat+beeplength
    stream[beat: offset] = (beep +stream[beat: offset])/2

scaled = np.int16(stream / np.max(np.abs(stream)) * 32767)
write('E:\musik\\trainingsmusic\\test.wav', rate, scaled)