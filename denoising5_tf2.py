#!/usr/bin/env python

# Test denoising convolutional autoencoder with Tensorflow2.0
# adding noise from data
# To train and plot
#  % ./denoising5_tf2.py 0 1
# To train without display (job)
#  %srun --cpus-per-task=10 ./denoising5_tf2.py 0 0
# To apply and plot
#  % ./denoising5_tf2.py 1 1 

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Waveform has 1024 sample-points
npoints = 1024 # 256 # number of sample-points to be used

load_weights = bool(int(sys.argv[1])) # True #False #True
plot_data = bool(int(sys.argv[2])) #True #False


filename = os.path.basename(__file__)
filename = os.path.splitext(filename)[0]


# Build model with functional API

input_img = Input(shape=(npoints,1))
x = Conv1D(64, 5, padding='same', activation='relu')(input_img)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(32, 5, padding='same', activation='relu')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(32, 5, padding='same', activation='relu')(x)
encoded = MaxPooling1D(2, padding='same')(x)

x = Conv1D(32, 5, padding='same', activation='relu')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(32, 5, padding='same', activation='relu')(x)
x = UpSampling1D(2)(x)
x = Conv1D(64, 5, padding='same', activation='relu')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 5, padding='same', activation='sigmoid')(x)

autoencoder = Model(inputs=input_img, outputs=decoded)

autoencoder.compile(optimizer='adam', loss='mse')
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy') 
autoencoder.summary()

# if plot_data:
#     #from keras.utils.vis_utils import plot_model
#     from tensorflow.python.keras.utils.vis_utils import plot_model
#     plot_model(autoencoder, to_file="architecture.png", show_shapes=True)



# Load dataset
#x_original = np.loadtxt('wf11000_ori.csv', delimiter=',')
x_original = pd.read_pickle('wf11100.pkl').to_numpy()
x_noise = pd.read_pickle('wf328469.pkl').to_numpy()

nsamples = min(len(x_original), len(x_noise))
x_original = x_original[0:nsamples]
x_noise = x_noise[0:nsamples]


# shape data so that value in [0, 1]
x_original = x_original.astype('float32')
x_original = x_original.T[-npoints:].T # keep last npoints
x_noise = x_noise.astype('float32')
x_noise = x_noise.T[-npoints:].T # keep last npoints


# Add noise
x_train_noisy = x_original + x_noise

scale = 5
offset = 0.05 # 50 mV
x_original *= scale # scale
x_original += offset * scale;
x_train_noisy *= scale # scale
x_train_noisy += offset * scale; # add 50 mV offset

x_original = np.clip(x_original, 0, 1);
x_train_noisy = np.clip(x_train_noisy, 0, 1);

x_original = np.reshape(x_original, (len(x_original), npoints, 1))
x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), npoints, 1))

print (x_original.shape)

history=[]
if not load_weights:

    # Callback for model checkpoints
    checkpoint = ModelCheckpoint(
        filepath = filename + ".h5",
        save_best_only=True)
    
    # 'labels' are the pictures themselves
    hist = autoencoder.fit(x_train_noisy, x_original,
                           epochs=20, #50,
                           batch_size=256,
                           shuffle=True,
                           validation_split=0.1,
                           callbacks=[checkpoint])


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
if plot_data:
    plt.plot(history['loss'], linewidth=3, label='train')
    plt.plot(history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-2, 0.1)
    plt.ylim(1e-5, 1e-3) #mse
    #plt.yscale('log')
    #plt.show()

    
# test data
x_test = x_original[-11:]
x_test_noisy = x_train_noisy[-11:]
decoded_imgs = autoencoder.predict(x_test_noisy)

# revert scale and offset
x_test -= scale * offset
x_test /= scale
x_test_noisy -= scale * offset
x_test_noisy /= scale
decoded_imgs -= scale * offset
decoded_imgs /= scale



if plot_data:
    # 何個表示するか
    n = 1
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # # オリジナルのテスト画像を表示
        # ax = plt.subplot(2, n, i+1)
        # plt.plot(x_test[i])
        # # ax.get_xaxis().set_visible(False)
        # # ax.get_yaxis().set_visible(False)
        
        # # 変換された画像を表示
        # ax = plt.subplot(2, n, i+1+n)
        # plt.plot(decoded_imgs[i]
        # # ax.get_xaxis().set_visible(False)
        # # ax.get_yaxis().set_visible(False)

        plt.plot(x_test[i], label="original")
        plt.plot(x_test_noisy[i], label="noisy")
        plt.plot(decoded_imgs[i], label="decoded")

    plt.legend()
    plt.show()
