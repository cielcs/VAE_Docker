import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from VAE_callbacks import VAECheckpoint

# GPUデバイスの取得
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # GPUが利用可能な場合、メモリの動的な確保を有効にする
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

VAEModulePath = '/content/VAE_movie'
validimgFolderPath = '/content/data/valid_data'

import sys
sys.path.append(VAEModulePath)

import VAE_movie

import ImageDataGenerator
# import importlib
# importlib.reload(VAE)
# importlib.reload(ImageDataGenerator)
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model

datagen = ImageDataGenerator.MovieImageDataGenerator()
datagen2 = ImageDataGenerator.MovieImageDataGenerator()
latent_dim = 16 #16通常
encoder = VAE_movie.create_encoder(latent_dim)
decoder = VAE_movie.create_decoder(latent_dim)

batch_size=32
valid_size=16
epochs=10000
steps_per_epoch=10#10
input_shape = (6,64,64,3)

restart = False

#Adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
if not restart:
        vae = VAE_movie.VAE_movie(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)) #learning_rateを変えた詳しくは6192new.txt
else:
        print("restart")

        vae = VAE_movie.VAE_movie(None, None)
        vae.load('models_movie/encoder.h5','models_movie/decoder.h5')
        vae.compile(optimizer=keras.optimizers.Adam())

train_generator=datagen.flow('/content/data/movies/train_1/mov', input_shape, batch_size=batch_size)
valid_generator=datagen2.flow('/content/data/movies/test/test2', input_shape, batch_size=valid_size)
#valid_generator=datagen.flow(imgFolderPath, input_shape, batch_size=valid_size, subset='validation')

os.makedirs('models', exist_ok=True)

# modelCheckpoint = ModelCheckpoint(filepath = '/models/vaemodel.h5', #'models_movie/vaemodel.h5'
#                                   #monitor='val_loss',
#                                   monitor='val_loss',
#                                   verbose=1,
#                                   save_best_only=False,
#                                   save_weights_only=False,
#                                   mode='min',#min
#                                   save_freq=1)

vaeCheckpoint = VAECheckpoint(encoder_path = 'models/encoder_{epoch}.h5', #'models_movie/vaemodel.h5'
                                decoder_path = 'models/decoder_{epoch}.h5',
                                  #monitor='val_loss',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_freq=1000,
                                  save_best_only=False,
                                  save_weights_only=False,
                                  mode='min'),#min
                                  


history_acc = vae.fit(train_generator, 
        batch_size=batch_size, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs,
        #validation_data=valid_generator,
        #validation_steps=10,
        callbacks=[vaeCheckpoint])
         
#

print(history_acc.history)
vae.save('models_movie/encoder.h5','models_movie/decoder.h5')