#!/usr/bin/env python

# A denoising convolutional autoencoder with Tensorflow2.0
# applied to waveform data.
#
# Noise from data is added to MC signal data.
# Data augmentation is applied.
# You need datasets of signal and noise, separately, in pickle format.
# To train and plot
#  % ./denoising6_tf2.py 0 1
# To train without display (job)
#  %srun --cpus-per-task=10 ./denoising6_tf2.py 0 0
# To apply and plot
#  % ./denoising6_tf2.py 1 1 


# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
# api_key and workspace are supposed to be set in .comet.config file,
# otherwise set here like Experiment(api_key="AAAXXX", workspace = "yyy", project_name="zzz")
experiment = Experiment(project_name="wf_denoising")

import os, sys
import numpy as np
import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# arg
load_weights = bool(int(sys.argv[1])) 
plot_data = bool(int(sys.argv[2])) 
filename = os.path.basename(__file__)
filename = os.path.splitext(filename)[0]

import matplotlib
if not plot_data:
    matplotlib.use("Agg") # this is necessary when using plt without display (batch)
import matplotlib.pyplot as plt


# Parameters

# Waveform has 1024 sample-points
npoints = 1024 # 256 # number of sample-points to be used
scale = 5
offset = 0.05 # 50 mV

signal_dataset_file = 'wf11100.pkl'
noise_dataset_file  = 'wf328469.pkl'

# basic hyper-parameters
params = {
    'optimizer':   'adam',
    'loss':        'mse', #'binary_crossentropy', 
    'epochs':      10, # 20,
    'batch_size':  256,
}
# additional parameters
params2 = {
    'conv_activation':     'relu',
    'output_activation':   'linear', #'sigmoid',
    'signal_dataset_file': signal_dataset_file,
    'noise_dataset_file':  noise_dataset_file,
    'npoints':             npoints,
    'scale':               scale,
    'offset':              offset,
}
experiment.log_parameters(params2)


# Build model with functional API

input_img = Input(shape=(npoints,1))
x = Conv1D(64, 5, padding='same', activation=params2['conv_activation'])(input_img)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(32, 5, padding='same', activation=params2['conv_activation'])(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(32, 5, padding='same', activation=params2['conv_activation'])(x)
encoded = MaxPooling1D(2, padding='same')(x)

x = Conv1D(32, 5, padding='same', activation=params2['conv_activation'])(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(32, 5, padding='same', activation=params2['conv_activation'])(x)
x = UpSampling1D(2)(x)
x = Conv1D(64, 5, padding='same', activation=params2['conv_activation'])(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 5, padding='same', activation=params2['output_activation'])(x)

autoencoder = Model(inputs=input_img, outputs=decoded)

autoencoder.compile(optimizer=params['optimizer'], loss=params['loss']) 
autoencoder.summary()



# Load dataset
#x_original = np.loadtxt('wf11000_ori.csv', delimiter=',')
x_original = pd.read_pickle(signal_dataset_file).to_numpy()
x_noise = pd.read_pickle(noise_dataset_file ).to_numpy()

nsamples = min(len(x_original), len(x_noise))
test_nsamples = (int)(nsamples * 0.1)

# divide into train and test samples
x_train, x_test = train_test_split(x_original, test_size=test_nsamples, random_state=1)
x_noise_train, x_noise_test = train_test_split(x_noise, test_size=test_nsamples, random_state=1)


class MySequence(tf.keras.utils.Sequence):
    """ My generator """

    def __init__(self, x_original, x_noise, batch_size,
                 npoints=1024,
                 scale=5,
                 offset=0.05,
                 signal_scale_range=0.0,
                 noise_scale_range=0.0,
                 baseline_range=0.0,
                 shift_range=0
                 ):
        """ 
        Constructor 
        arguments:
           signal_scale_range: range of signal scaling
        """
        self.x = x_original
        self.noise = x_noise
        self.batch_size = batch_size
        self.batches_per_epoch = int((len(self.x) - 1) / self.batch_size) + 1
        self.npoints = npoints
        self.scale = scale
        self.offset = offset

        # parameters for data augmentation
        self.signal_scale_range = signal_scale_range
        self.noise_scale_range = noise_scale_range
        self.baseline_range = baseline_range        
        self.shift_range = shift_range

    def __getitem__(self, idx):
        """ Generate the batch data """

        x_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_size = len(x_batch)

        # rotate noise array
        noise_batch = np.roll(self.noise, -(idx * self.batch_size) % len(self.noise));
        noise_batch = noise_batch[0:batch_size] 

        # shape data 
        x_batch, noisy_batch = self.preprocess(x_batch, noise_batch)

        return noisy_batch, x_batch

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass


    def preprocess(self, x_original, x_noise):
        """ 
        shape raw data for input 
        apply data augmentation
        """

        # Shape data in appropriate format 
        x_original = x_original.astype('float32')
        x_original = x_original.T[-self.npoints:].T # keep last npoints
        x_noise = x_noise.astype('float32')
        x_noise = x_noise.T[-self.npoints:].T # keep last npoints

        # Data augmentation
        if self.signal_scale_range > 1.0:
            x_original = x_original * random.uniform(1/self.signal_scale_range, self.signal_scale_range)

        if self.noise_scale_range > 1.0:
            x_noise = x_noise * random.uniform(1/self.noise_scale_range, self.noise_scale_range)

        if self.baseline_range > 0.0:
            x_noise = x_noise + random.uniform(-self.baseline_range, self.baseline_range)
                                                     

        # Add noise
        x_noisy = x_original + x_noise

        # Adjust scale and offset of waveforms
        x_original *= scale # scale
        x_original += offset * scale;
        x_noisy *= scale # scale
        x_noisy += offset * scale; # add 50 mV offset


        # Values in [0,1]
        x_original = np.clip(x_original, 0, 1);
        x_noisy = np.clip(x_noisy, 0, 1);

        # To match the input shape for Conv1D with 1 channel
        x_original = np.reshape(x_original, (len(x_original), self.npoints, 1))
        x_noisy = np.reshape(x_noisy, (len(x_noisy), self.npoints, 1))
 

        return x_original, x_noisy



history=[]
if not load_weights:
    # generators
    train_batch_generator = MySequence(x_train, x_noise, params['batch_size'], 
                                       scale=scale, offset=offset)
    test_batch_generator = MySequence(x_test, x_noise_test, params['batch_size'], 
                                      scale=scale, offset=offset)

    # Callback for model checkpoints
    checkpoint = ModelCheckpoint(
        filepath = filename + "-{epoch:02d}.h5",
        save_best_only=True)
    
    # 'labels' are the pictures themselves
    hist = autoencoder.fit(train_batch_generator,
                           epochs=2, #50,
                           steps_per_epoch=train_batch_generator.batches_per_epoch,
                           shuffle=True,
                           validation_data=test_batch_generator,
                           validation_steps=test_batch_generator.batches_per_epoch
                           ,callbacks=[checkpoint])


    # Save history
    with open(filename + '_hist.json', 'w') as f:
        json.dump(hist.history, f)
    history = hist.history
        
    # Save the weights
    autoencoder.save_weights(filename + '_weights.h5')
else:
    # Load weights
    autoencoder.load_weights(f'{filename}_weights.h5')

    # Load history
    with open(f'{filename}_hist.json', 'r') as f:
        history = json.load(f)

    autoencoder.save(filename + '.h5', include_optimizer=False)
        
# Plot training history 
plt.plot(history['loss'], linewidth=3, label='train')
plt.plot(history['val_loss'], linewidth=3, label='valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-2, 0.1)
plt.ylim(1e-5, 1e-3) #mse

    
# test data
x_test = x_original[-11:]
x_noise_test = x_noise_train[-11:]
x_test, x_test_noisy = test_batch_generator.preprocess(x_test, x_noise_test)
decoded_imgs = autoencoder.predict(x_test_noisy)

# revert scale and offset
x_test -= scale * offset
x_test /= scale
x_test_noisy -= scale * offset
x_test_noisy /= scale
decoded_imgs -= scale * offset
decoded_imgs /= scale


# How many waveforms to be displayed
n = 1
plt.figure(figsize=(20, 6))
for i in range(n):
    plt.plot(x_test[i], label="original")
    plt.plot(x_test_noisy[i], label="noisy")
    plt.plot(decoded_imgs[i], label="decoded")
    plt.legend()

# Send this plot to comet
experiment.log_figure(figure=plt)

if plot_data:
    plt.show()
