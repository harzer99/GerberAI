import numpy as np
import time
import scipy as sc
import matplotlib.pyplot as plt
from scipy import signal

sample = np.load("E:\musik\\trainingsmusic\\sample.npy")
sr = 22050
window = 1
starttime = 200
sample = sample[starttime*sr:window*sr+starttime*sr]
#sample = np.zeros(22050*window)+1
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


lnmax = np.log10(sr)
resolution = 1000
space = np.logspace(1.1, lnmax, num = resolution).astype(int)
#print(space)
frequency_bins = np.zeros(resolution)

for i in range(len(space)-1):
    if space[i] > sr:
        break
    frequency_bins[i] = np.average(realfourier[space[i]:space[i+1]+1])

resolution = 500
space2 = np.logspace(1.1, lnmax, num = resolution).astype(int)
frequency_bins2 = np.zeros(resolution)

t_1 = time.time()
realfourier = np.abs(np.fft.fft(sample))
for i in range(len(space2)-1):
    #fouriersnippet = realfourier[space2[i]:space2[i+1]+1]
    frequency_bins2[i] = np.average(realfourier[space2[i]:space2[i+1]+1])
t_2 =time.time()
print('rfft and downsampling {}'.format(t_2-t_1))



plt.plot(frequency_bins)
plt.plot(frequency_bins2)
#plt.plot(realfourier)
plt.show()