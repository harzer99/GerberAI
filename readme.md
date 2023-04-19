# Overview
GerberAI is a program generates an image from a short audio input. It uses Torchopenl3 to embedd the audio data and then generates an image via a custom trained GAN. It also predicts the next beat and pushes the image in sync with it. 

# Requirements
A Nvidia GPU is suggested. A fast CPU might just be enough though.

# Running the program
You first need to modify the presenter.py:
You need change the path to the folder which contain the generators. In the current configuration the generators need to be saved as generator_*.pt. These generators will be loaded into system memory alphabetically and the presenter will do a random walk through that list while it is running. You might also need to change x and y shift for the presentation window depending on the screen setup you have.
