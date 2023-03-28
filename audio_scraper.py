import numpy as np
import pytube as pt
import librosa
import os
from tqdm import tqdm
from multiprocessing.pool import Pool


class Worker:
    def __init__(self, worker, n):
        self.n = n
        self.worker = worker

    def __len__(self):
        return self.n

    def __iter__(self):
        return next(worker)


def single_track_download(video):
    try:
        stream = video.streams.filter(only_audio=True).order_by('abr').last()   #get the audio stream with the highest bitrate
        stream.download(output_path= output_path)
    except:
        print('skipping song')

class audio_scraper():
    def __init__(self, track_directory, output_directory, training_window) -> None:
        self.track_directory = track_directory
        self.training_window = training_window
        self.output_directory = output_directory
    
    def download(self, playlist_url):
        print('Collecting playlist')
        playlist = pt.Playlist(playlist_url)
        n = len(playlist)
        print('Number of Songs: {:.2f}'.format(n))
        i = 0
        pool = Pool()
        worker = pool.imap_unordered(single_track_download, playlist.videos)
        #tqdm(Worker(worker, n))
        with tqdm(total=n) as pbar:
            while i < n:
                next(worker)
                #print(f'Song {i}/{n} got downloaded')
                pbar.update()
                i += 1
                #print('Download')

        
        #for video in tqdm(playlist.videos, desc='Downloading tracks'):
            #stream = video.streams.filter(only_audio=True).order_by('abr').last()   #get the audio stream with the highest bitrate
            #stream.download(output_path= self.track_directory)

    def analyze(self):
        audio = np.array([], dtype  = 'float32')
        flags = np.array([], dtype = 'int64')
        for filename in tqdm(os.listdir(self.track_directory), desc='rewriting to trainingdata'):
            path = os.path.join(self.track_directory, filename)
            y, sr = librosa.load(path)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm = 120)
            beats = np.delete(beats, np.where(beats < self.training_window*sr))     #getting rid of the beats that would result in training on parts of the previous track
            audio = np.append(audio, y)
            flags = np.append(flags, beats + len(audio))
        np.save(os.path.join(self.output_directory, 'audio'), audio)
        np.save(os.path.join(self.output_directory, 'flags'), flags)
        
#playlist_url = 'https://www.youtube.com/watch?v=3V_yKgv1UUU&list=PL1ntfXx-b2MyxKr-kRkpe0xxcqwB4fRLz&ab_channel=HATE'
playlist_url = 'https://www.youtube.com/playlist?list=PLHZZwMs7hXSDFqLPM4q5CaUwy5r4B5CS7'
output_path = 'training_data/raw_music'

if __name__ == '__main__':
    myaudioanalyzer = audio_scraper('training_data/raw_music', 'training_data', 6)
    myaudioanalyzer.download(playlist_url)
    myaudioanalyzer.analyze()
