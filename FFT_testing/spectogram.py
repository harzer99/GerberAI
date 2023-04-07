import numpy as np

class Spectogram():
    def __init__(self, auduio_in, beat_flags, outdir, sr, samplelength):
        self.audio_in = auduio_in
        self.beat_flags = beat_flags
        self.outdir = outdir
        self.sr = sr
        self.samplelength = samplelength

    def create_spectogram(sample):
        realfourier = np.abs(np.fft.rfft(sample))
        
