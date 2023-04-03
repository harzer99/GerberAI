import numpy as np
import time
import scipy as sc
import matplotlib.pyplot as plt


sample = np.load("E:\musik\\trainingsmusic\\sample.npy")
sr = 22050
window = 6
starttime = 20
sample = sample[starttime*sr:window*sr+starttime*sr]
t_1 = time.time()
np.fft.fft(sample)
t_2 =time.time()
print('np_fft runtime {}'.format(t_2-t_1))

t_1 = time.time()
realfourier = np.abs(np.fft.fft(sample))
t_2 =time.time()
print('np_rfft runtime {}'.format(t_2-t_1))

t_1 = time.time()
welch_spectrum = sc.signal.welch(sample)
t_2 =time.time()
print('sc_welch runtime {}'.format(t_2-t_1))

t_1 = time.time()
reikna_fft = reikna_FFT.FFT(sample)
t_2 =time.time()
print('sc_welch runtime {}'.format(t_2-t_1))
