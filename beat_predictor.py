# Beat tracking example
import librosa
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time

class Beat_detector:
    def __init__ (self, sr, update_time, analyze_time):
        self.sr = sr
        self.update_time = update_time
        self.analyze_time = analyze_time
        self.futurebeats = np.zeros(3)
        self.analyze(self.sr, self.analyze_time, self.update_time) 

    def get_device_name(self):
        "Uses sounddevice to get the name of the mic in case it changed"
        s = sd.query_devices()
        device = f"""{s[1]["name"]})"""
        return device
        

    def analyze(self, sr, analyze_time, update_time):
        chunk = int(update_time*sr)
        audiomemory = np.zeros(int(sr*analyze_time))
        while True:
            t_1 = time.time()
            frames = sd.rec(int(update_time*sr), channels  = 2, samplerate=sr)[:,1]
            tempo, beats = librosa.beat.beat_track(y=audiomemory, sr=sr, start_bpm = 120)
            meanbeat = np.mean(librosa.frames_to_time(beats, sr=sr))

            if tempo == 0:
                tau = 10
            else: 
                tau = 1/tempo*60
            t_3 = time.time()
            print(t_1)
            if len(beats)%2:                                                
                n_a = np.ceil((t_3-meanbeat-t_1)/2)                         #estimate how many beats ahead we are from meanbeat
                for i in range(3):
                    self.futurebeats[i] = t_1+meanbeat+tau*(n_a+i)          #calculating times of future beats
            else:
                n_a = np.ceil((t_3-meanbeat-t_1)/2)
                for i in range(3):
                    self.futurebeats[i] = t_1+meanbeat+tau*(n_a+i-1/2)      #in the even case the mean beat is offset by half a beat timing

            runtime = time.time()-t_1
            sd.wait()
            
            audiomemory = np.append(audiomemory[chunk:], frames)
            print('Estimated tempo: {:.2f} bpm, runtime:{:.2f} '.format(tempo, runtime,))
            print(self.futurebeats)
    
sr=5500
update_time = 1.0
analyze_time = 20.0

analyzer = Beat_detector(sr, update_time, analyze_time)

print(analyzer.futurebeats)


