# Beat tracking example
import librosa
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import os

os.chdir(os.getcwd())

sr = 5500
RECORD_SECONDS = 2
y, sr = librosa.load('pathtoaudiofile')
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
#calculating a tempo estimate every record_second intervall. 
np.save('audio', y)
np.save('flags', beats)
    
    


# 3. Run the default beat tracker

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
print(beats)

# 4. Convert the frame indices of beat events into timestamps
