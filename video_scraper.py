import numpy as np
import os
import subprocess
import math
import json
import cv2
import scipy.io.wavfile
import matplotlib.pyplot as plt
import pytube as pt
from multiprocessing.pool import Pool
from tqdm import tqdm
import multiprocessing

stills_path = 'videolists\\stills.txt'
motions_path = 'videolists\\motionvids.txt'

open(stills_path, 'a')
open(motions_path, 'a')
with open(stills_path, 'r') as out:
    still_videos = [x.strip() for x in out]

with open(motions_path, 'r') as out:
    motion_videos = [x.strip() for x in out]

def get_frames(seek, video_stream):
        process = multiprocessing.current_process()
        pid = process.pid
        duration = 5
        filename = 'videocache\\download{}.mkv'.format(pid)
        subprocess.run(['ffmpeg', '-y', '-ss', f'{seek:.2f}', '-i', video_stream, '-t', f'{duration:.2f}', '-c:v', 'copy', filename]);
        print('got video download')
        
        frames = []
        video = cv2.VideoCapture(filename)
        i = 0
        while video.isOpened() and i < duration*round(video.get(cv2.CAP_PROP_FPS)):
            i+=1
            ret, frame = video.read()
            if ret == False:
                break
            frames.append(frame)
        
        #os.remove(filename)
        #subprocess.run(['ffmpeg', '-y', '-ss', f'{seek:.2f}', '-i', audio_stream, '-t', f'{duration:.2f}', 'download.wav']);
        #print('got audio download')
        video.release()
        os.remove(filename)
        return np.array(frames)
    
def filter_motionvids(url):

    if url in still_videos or url in motion_videos:
        print('video already covered')
        return
    try:
        frame = json.loads(subprocess.run(["yt-dlp", url, '-j', '--skip-download'], capture_output=True, text=True).stdout)
        duration = frame['duration']

        video_stream, audio_stream = subprocess.run(["yt-dlp", url, '-g', '-f', 'worst[vcodec!=none][ext!=3gp],worst[acodec!=none][ext!=3gp]'], capture_output=True, text=True).stdout.split('\n')[:2]
        #print(f'open stream for {url}')

        for i in range(1):
            seek = np.random.uniform(low = 5, high = duration-5)
            frames= get_frames(seek, video_stream)

            if np.mean(np.abs(frames[-1]-frames[0])) < 40:
                print('still image')
                with open(stills_path, 'a') as out:
                        print(url, file=out)
                
                
            else:
                with open(motions_path, 'a') as out:
                        print(url, file=out) 
    except:
         print('download failed')
    
            

class Video_Scraper():
    def __init__(self, playlisturl) -> None:
        self.playlisturl = playlisturl
        self.p = pt.Playlist(playlisturl)
        self.yt_urls = self.p.video_urls
        

    def run(self):
        n = len(self.yt_urls)-1
        i = 0
        pool = Pool()
        worker = pool.imap_unordered(filter_motionvids, self.yt_urls)
        with tqdm(total=n) as pbar:
            while i < n:
                next(worker)
                pbar.update()
                i += 1
        
        
                  

    
                
if __name__ == '__main__':
    playlisturl = ''
    myvideoscraper = Video_Scraper(playlisturl)
    myvideoscraper.run()