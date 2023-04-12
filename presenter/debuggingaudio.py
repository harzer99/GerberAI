import os
os.environ['PY_SSIZE_T_CLEAN'] = '1'
import pyaudio
import time

# Open a new pulseaudio stream to capture system audio output
pa = pyaudio.PyAudio()
stream = pa.open(format=pa.get_format_from_width(2),
                 channels=2,
                 rate=44100,
                 input=True,
                 input_device_index=2)

print(pa.get_default_output_device_info()['index'])
for ii in range(pa.get_device_count()):
    print(ii, pa.get_device_info_by_index(ii).get('name'))
# Play the captured audio using pyaudio
p = pyaudio.PyAudio()
output_stream = p.open(format=pa.get_format_from_width(2),
                       channels=2,
                       rate=44100,
                       output=True)
while True:
    audio_data = stream.read(1024)
    output_stream.write(audio_data)