import numpy as np
import pytube as pt
import librosa
import os
from tqdm import tqdm
from multiprocessing.pool import Pool
import subprocess
from functools import partial


def single_video_download(video_url):
    try:
        vid_id = pt.extract.video_id(video_url) 
        video = pt.YouTube(video_url)
        stream = video.streams.filter(res="480p", only_video=True).first()   #get the audio stream with the highest bitrate
        A = stream.download(output_path= video_directory, filename=vid_id+'.mp4')
        stream = video.streams.filter(only_audio=True).order_by('abr').last()   #get the audio stream with the highest bitrate
        B = stream.download(output_path= audio_directory, filename=vid_id+'.webm')

        # double checking if both files have sucsessfully been created
        if not (os.path.exists(A) and os.path.exists(B)):
            if os.path.exists(A):
                os.remove(A)
            if os.path.exists(B):
                os.remove(B)
            print('one download failed')
        return
    except:
        print('skipping song')


def export_frame(beat, sr, vidfile, audio):
    seek = beat/sr
    image_path = f'{image_directory}\\{beat + len(audio)}.png'
    subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-y', '-ss', f'{seek:.2f}', '-i', vidfile, '-frames:v', '1', image_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);


#exporting frames and audio snippeds for the pretrained GAN
def export_for_imgen(audio_file):
    audio_path = audio_directory + '\\'+ audio_file
    id = audio_path.split('\\')[-1]
    id = id[0:-5]
    video_path = video_directory +'\\' + id + '.mp4'
    id_vid = video_path.split('\\')[-1]
    id_vid = id_vid[0:-4]
    assert id == id_vid

    if not os.path.isfile(video_path):
        print('no vid to this audio')
        return

    #getting length of video
    args= ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    length = float(result.stdout)
    if length > 600:
        print('skipped track due to being more than 10min')
        return
    
    # running through each video and the corresponding audio to create snippets
    i = 0
    seeks = np.linspace(start = 10, stop = length-10, num = 100)
    for seek in seeks:
        path = snippet_directory + '\\'+ id + '_'+ str(i).zfill(3)
        image_path = path + '.png'
        snipped_path = path + '.wav'
        CMD_audio = ['ffmpeg', '-hide_banner', '-nostats', '-y', '-ss', f'{seek-5:.2f}', '-i', audio_path, '-t', '5', '-ar', '44100',  snipped_path]
        CMD_image = ['ffmpeg', '-hide_banner', '-nostats', '-y', '-ss', f'{seek:.2f}', '-i', video_path, '-frames:v', '1', image_path]
        subprocess.run(CMD_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);
        subprocess.run(CMD_image, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);
        if os.path.getsize(snipped_path) < 1000:
            os.remove(image_path)
            os.remove(snipped_path)
            print('something went wrong with the audio, deleting file again')
        i += 1

class Video_Prepper():
    def __init__(self, video_directory, audio_directory, output_directory, training_window) -> None:
        self.video_directory = video_directory
        self.audio_directory = audio_directory
        self.training_window = training_window
        self.output_directory = output_directory
    
    def download(self, playlist_file):
        print('Collecting playlist')
        with open(playlist_file, 'r') as out:
            playlist = [x.strip() for x in out]
        n = len(playlist)
        print('Number of Songs: {:.2f}'.format(n))
        i = 0
        pool = Pool()
        worker = pool.imap_unordered(single_video_download, playlist)
        #tqdm(Worker(worker, n))
        with tqdm(total=n) as pbar:
            while i < n:
                next(worker)
                #print(f'Song {i}/{n} got downloaded')
                pbar.update()
                i += 1
                #print('Download')

    def analyze(self):
        audio = np.array([], dtype  = 'float32')
        flags = np.array([], dtype = 'int64')
        for filename in tqdm(os.listdir(self.audio_directory), desc='rewriting to trainingdata'):
            path = os.path.join(self.audio_directory, filename)
            y, sr = librosa.load(path)
            y = librosa.to_mono(y)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm = 120)
            beats = librosa.frames_to_time(beats, sr=sr)
            beats = (beats*sr).astype(int)
            beats = np.delete(beats, np.where(beats < self.training_window*sr))     #getting rid of the beats that would result in training on parts of the previous track
            if sr != 22050:
                print('differing samplerate')
            vidfile = video_directory+'\\'+filename[:-4]+'mp4'

            #multithreading the frame export
            n = len(beats)
            i = 0
            pool = Pool()
            partial_export = partial(export_frame, sr = sr, vidfile = vidfile, audio = audio)
            worker = pool.imap_unordered(partial_export, beats)
            while i < n:
                next(worker)
                i += 1
            
            flags = np.append(flags, beats + len(audio))
            audio = np.append(audio, y)
            print(np.max(flags), len(audio))
        np.save(os.path.join(self.output_directory, 'audio'), audio)
        np.save(os.path.join(self.output_directory, 'flags'), flags)
        print(sr)
    
    # exports snippeds from downloaded .mp4 and corresponding .webm files from youtube downlaods
    def export_snippets(self):
        audio_files = os.listdir(self.audio_directory)    
        n = len(audio_files)
        print('Number of Songs: {:.2f}'.format(n))
        i = 0
        pool = Pool()
        worker = pool.imap_unordered(export_for_imgen, audio_files[:n])
        with tqdm(total=n) as pbar:
            while i < n:
                next(worker)
                pbar.update()
                i += 1
        pool.close() # Close the pool to prevent any more tasks from being submitted
        pool.join()
        print('finished')
        
###########################################################
#these two paths have to be added. It will create 
###########################################################

playlist_file = 'youtubescraper\\videolists\\hate_podcast.txt'
main_directory = 'I:\\GerberAI'


###########################################################
#creating all the directories
video_directory = main_directory +'\\downloaded_videos'
audio_directory = main_directory +'\\downloaded_audios'
image_directory = main_directory + '\\images'
snippet_directory = main_directory +'\\snippets'
imgen_output = main_directory +'\\imggen_output'

os.makedirs(video_directory, exist_ok = True )
os.makedirs(audio_directory, exist_ok = True)
os.makedirs(image_directory, exist_ok = True)
os.makedirs(snippet_directory, exist_ok = True)


if __name__ == '__main__':
    myprepper = Video_Prepper(video_directory, audio_directory, main_directory, 6)
    #myprepper.download(playlist_file)
    #myprepper.analyze()
    myprepper.export_snippets()