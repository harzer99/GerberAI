import csv
import os

class Directories:
    def __init__(self, paths_csv = 'presenter/paths.csv'):
        self.paths_csv = paths_csv
        self.directories = self.create_subdirectories()
    def create_subdirectories(self):
        with open(self.paths_csv, newline='') as csvfile:
            csv_reader = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
            main_directory = csv_reader[0][1]
        
        subdirectories =['videolists', 
                        'downloaded_audios', 
                        'downloaded_videos', 
                        'snippets',
                        'imggen_output', 
                        os.path.join('imggen_output', 'checkpoints'), 
                        os.path.join('imggen_output','images'),                     
                        os.path.join('imggen_output', 'plots')
        ]
        labels = [  'videolists', 
                    'downloaded_audios', 
                    'downloaded_videos', 
                    'snippets', 
                    'imggen_output',
                    'checkpoints', 
                    'images',                     
                    'plots'
        ]
        direct_dict = {}
        i = 0
        for subdirectory in subdirectories:
            direct_dict[labels[i]] = os.path.join(main_directory, subdirectory)
            os.makedirs(direct_dict[labels[i]], exist_ok = True)
            i+=1
        direct_dict['main']=main_directory
        return direct_dict
    
    
if __name__ =='__main__':
    my_directories = Directories()
    directories = my_directories.directories