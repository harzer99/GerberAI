
from main import *
from torch import Tensor
import scipy

timing = 132300
sr = 22500
time = 800*sr

def get_sample(array, model: Mood, time=time):
    music = Tensor(array[time: time + timing].reshape(1, 1, -1)).to(device)
    music = model.preprocess(music)
    processed = model(music)
    return processed.cpu().detach().numpy().reshape(-1), music.cpu().detach().numpy().reshape(-1)


if __name__ == '__main__':
    m = torch.load('autoencoder\\new_2-2')
    

    some_array = np.load('E:\musik\\trainingsmusic\\audio.npy')

    sample, presample = get_sample(some_array, m)
    print(sample)
    scipy.io.wavfile.write('audio_out.wav', 5512, sample)
    scipy.io.wavfile.write('audio_pre.wav', 5512, presample)
    original = some_array[time:time+timing]
    scipy.io.wavfile.write('audio_in.wav', 22000, original)

