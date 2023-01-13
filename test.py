import numpy as np
import madmom
from madmom.models import BEATS_LSTM
from matplotlib import pyplot as plt

proc = madmom.features.beats.RNNBeatProcessor(online=True, nn_files=[BEATS_LSTM[0]])
madmom.features.tempo.TempoHistogramProcessor(60, 180)
activations = proc("E:\musik\Techno playlist youtube\Rennie Foster - Cherriep - Artificial Intelligence - RF060.mp3")
histogram = madmom.features.tempo.interval_histogram_acf(activations, min_tau=1, max_tau=None)
tempo = madmom.features.tempo.detect_tempo(histogram, 100)
tempo = np.transpose(tempo)
tempoweights = tempo[1]
i = np.argmax(tempoweights)
print(tempo[0,i])

plt.plot(tempo[0], tempo[1], marker='o')
plt.xlim(0, 300)
plt.show()