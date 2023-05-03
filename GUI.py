import customtkinter as ctk
import tkinter as tk
import sys
import os
import csv
from presenter import create_directories
from youtubescraper import video_scraper
from youtubescraper import video_download_analzye
from imggen import prepare_dataset
from imggen.train import train

class GUI():
    def __init__(self, app):
        ctk.set_appearance_mode("Dark")  # Modes: system (default), light, dark
        ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
        app.geometry("1000x300")
        app.grid_rowconfigure(0, weight=1)  # configure grid system
        app.grid_columnconfigure(0, weight=1)

        self.directories = create_directories.Directories().directories
        open(os.path.join(self.directories['videolists'], 'playlist_url.txt'), 'a')
        with open(os.path.join(self.directories['videolists'], 'playlist_url.txt'), 'r') as f:
            self.playlist  = f.readlines()
            f.close()

            
        
        with open('presenter/paths.csv', newline='') as csvfile:
            rows = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
            data_directory = rows[0][1]

        self.label_data = ctk.CTkLabel(master=app, text='data directory')
        self.label_data.grid(row = 0, column = 0, padx=20, pady=20)
        self.label_directory = ctk.CTkLabel(master=app, text=data_directory)
        self.label_directory.grid(row = 0, column = 1, padx=20, pady=20)
        self.button_directory = ctk.CTkButton(master=app, text="browse", command=self.browse_data_directory)
        self.button_directory.grid(row = 0, column = 2, padx=20, pady=20)
        self.label_playlist= ctk.CTkLabel(master=app, text='playlist url')
        self.label_playlist.grid(row = 1, column = 0, padx=20, pady=20)
        self.entry_playlist = ctk.CTkEntry(app, width=500, placeholder_text = self.playlist)
        self.entry_playlist.grid(row = 1, column = 1, padx=20, pady=20)
        self.button_prepare = ctk.CTkButton(master=app, text="prepare data", command=self.prepare_data)
        self.button_prepare.grid(row = 3, column = 1, padx=20, pady=20)
        self.button_scraper = ctk.CTkButton(master=app, text="scrape vids", command=self.run_video_scraper)
        self.button_scraper.grid(row = 2, column = 2, padx=20, pady=20)
        self.button_scraper = ctk.CTkButton(master=app, text="download", command=self.download)
        self.button_scraper.grid(row = 3, column = 2, padx=20, pady=20)
        self.button_scraper = ctk.CTkButton(master=app, text="export snippets", command=self.export_snippets)
        self.button_scraper.grid(row = 4, column = 2, padx=20, pady=20)
        self.button_embedder = ctk.CTkButton(master=app, text="embedd", command=self.embedd)
        self.button_embedder.grid(row = 5, column = 2, padx=20, pady=20)
        self.button_train = ctk.CTkButton(master=app, text="train model", command=self.train_model)
        self.button_train.grid(row = 6, column = 1, padx=20, pady=20)
    #changes the data director and also creates all the subfolders
    def browse_data_directory(self):
        data_directory  = tk.filedialog.askdirectory()
        self.change_data_directory(data_directory)

    def change_data_directory(self, data_directory):
        if data_directory == '':
            return
        with open('presenter/paths.csv', newline='') as csvfile:
            rows = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
        rows[0][1]=data_directory
        with open('presenter/paths.csv', mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in rows:
                csv_writer.writerow(row)
        self.label_directory.configure(text = data_directory)
        self.directories = create_directories.Directories().directories

    #updates playlilst_url.txt file wit user input
    def update_playlist(self):
        if self.entry_playlist.get() !='':
            self.playlist = self.entry_playlist.get()
            with open(os.path.join(self.directories['videolists'],'playlist_url.txt'), 'w') as f:
                f.write(self.playlist)
                f.close
    
    def run_video_scraper(self):
        self.update_playlist()
        self.playlisturl = open(os.path.join(self.directories['videolists'],'playlist_url.txt'), 'r').read()
        self.myvideoscraper = video_scraper.Video_Scraper(self.playlisturl)
        self.myvideoscraper.run()

    #using Video_prepper class to download all youtube videos
    def download(self):
        myprepper = video_download_analzye.Video_Prepper(self.directories['downloaded_videos'], self.directories['downloaded_audios'], self.directories['main'], 6)
        myprepper.download(os.path.join(self.directories['videolists'], 'motionvids.txt'))
    
    def export_snippets(self):
        myprepper = video_download_analzye.Video_Prepper(self.directories['downloaded_videos'], self.directories['downloaded_audios'], self.directories['main'], 6)
        myprepper.export_snippets()

    def embedd(self):
        prepare_dataset.embedd_dataset(self.directories['snippets'], self.directories['main']+'/dataset.pt')

    def prepare_data(self):
        self.run_video_scraper()
        self.download()
        self.export_snippets()
        self.embedd()
    
    def train_model(self):
        train(self.directories['main']+'/dataset.pt', self.directories['imggen_output'])

# Use CTkButton instead of tkinter Button
if __name__ == '__main__':
    app = ctk.CTk()  # create CTk window like you do with the Tk window
    my_gui = GUI(app)
    app.mainloop()
