# machine-auscultation-classification
This project outlines two different neural network models to analyze and classify acoustic signal emanating from machines: (1) a backpropagation (BP) neural network; and (2) a convolutional neural network (CNN). For BP neural network, 13 features of sound are selected as input for the network. The input for CNN is Mel-spectrogram figure of sound and they are converted to binary format for training. The relative data pre-processing codes are in Data_Preprocessing folder, the network codes are in network model folder.

Sound_data folder which has two parts: chatter sound data and feedrate_speed_cutdepth sound data. 

In chatter sound data, there are 27 recording folder and corresponding label for each one is stored in mat file. 0 represents normal sound, 1 menas chatter.

In feedrate_speed_cutdepth sound data, the first value is feed rate, second value is speed and third value is cutdepth for each wav file.

# Requirements
Tensorflow 1.5.0

Python 3.6

Numpy
