import numpy as np
import pytube as pt
import librosa
import os
from tqdm import tqdm
from multiprocessing.pool import Pool

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
            try:
                path = os.path.join(self.track_directory, filename)
                y, sr = librosa.load(path)
                y = librosa.to_mono(y)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm = 120)
                beats = librosa.frames_to_time(beats, sr=sr)
                beats = (beats*sr).astype(int)
                beats = np.delete(beats, np.where(beats < self.training_window*sr))     #getting rid of the beats that would result in training on parts of the previous track
                flags = np.append(flags, beats + len(audio))
                audio = np.append(audio, y)
                print(np.max(flags), len(audio))
                if sr != 22050:
                    print('differing samplerate')
            except: 
                print('analyze error, skipping file')
        np.save(os.path.join(self.output_directory, 'audio'), audio)
        np.save(os.path.join(self.output_directory, 'flags'), flags)
        print(sr)

        
playlist_url = ''
output_path = 'E:\musik\\trainingsmusic\\techno'

if __name__ == '__main__':
    myaudioanalyzer = audio_scraper(output_path, 'E:\musik\\trainingsmusic', 6)
    #myaudioanalyzer.download(playlist_url)
    myaudioanalyzer.analyze()
