
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali

audio_data = np.array(y, dtype=np.float32)

@pipeline_def
def spectrogram_pipe(nfft, window_length, window_step, device='cpu'):
    audio = types.Constant(device=device, value=audio_data)
    spectrogram = fn.spectrogram(audio, device=device, nfft=nfft,
                                 window_length=window_length,
                                 window_step=window_step)
    return spectrogram