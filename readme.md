# Overview
GerberAI is a program generates an image from a short audio input. It uses Torchopenl3 to embed the audio data and then generates an image via a custom trained GAN. It also predicts the next beat and pushes the image in sync with it. 

# Requirements
A Nvidia GPU is suggested. A fast CPU might just be enough though.
We recommend installing Anaconda and creating a new environment.
First install pytorch and Cuda:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Then run the requirements.txt file in the root directory.

# Running the program
You first need to modify the presenter.py:
You need change the path to the folder which contain the generators. In the current configuration the generators need to be saved as generator_*.pt. These generators will be loaded into system memory alphabetically and the presenter will do a random walk through that list while it is running. You might also need to change x and y shift for the presentation window depending on the screen setup you have.

# Training
## Gathering the data
Current workflow is via youtube. First you need a youtube playlist that potentially contains music videos. Now put that link into the youtubescrapoer/video_scraper.py script. It will go trougth check each video if it is a still or a motion video. The links to the motion videos are saved in a file. When that is finished modify the youtubescraper/video_analyze script. You only need to add the path to the video links file and the main output directory. It will first download the whole video and audio streams and then cut them into snippets. Next you need open imggen/prepare_dataset.py and change the input and output paths. The script will embedd the audio snippets, downsample the images and save them as a pytorch file. This is your training data. 

## Training the model
For training open the imggen/train.py script. Change the paths for training data and output folder. Then run it. It will create sample images, checkpoints of the generator and discriminator and plots of the lossfunctions at given intervals.
