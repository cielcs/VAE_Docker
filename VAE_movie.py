"""
c.f. https://keras.io/examples/generative/vae/
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        sigma = tf.math.exp(0.5*z_log_var)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0., stddev=1.)
#        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_mean + sigma * epsilon

## Encoder
def create_encoder(latent_dim):
  input_shape = (6,64,64,3)
  encoder_inputs = keras.Input(shape=input_shape)
  x = layers.Conv3D(30, 3, padding="same")(encoder_inputs) 
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv3D(64, 3, strides=2, padding="same")(x)#64
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv3D(64, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv3D(64, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  x = layers.Flatten()(x)

  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_sigma")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  encoder.summary()
  return encoder

## Decoder
def create_decoder(latent_dim):
  encode_activation_shape = (3,32,32,64)

  latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')

  x = layers.Dense(encode_activation_shape[0]*encode_activation_shape[1]*encode_activation_shape[2]*encode_activation_shape[3])(latent_inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Reshape(encode_activation_shape)(x)
  x = layers.Conv3DTranspose(64, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv3DTranspose(64, 3, strides=2, padding="same")(x)#64
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv3DTranspose(30, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  x1 = layers.Conv3DTranspose(3, 3, padding="same")(x)
  x1 = layers.BatchNormalization()(x1)
  outputs_mu = layers.Activation('sigmoid')(x1)

  x2 = layers.Conv3DTranspose(3, 3, padding="same")(x)
  x2 = layers.BatchNormalization()(x2)
  outputs_log_var = layers.Activation('sigmoid')(x2)

  #decoder_outputs = layers.Activation('sigmoid')(x)

  decoder = keras.Model(latent_inputs, [outputs_mu, outputs_log_var], name="decoder")
  decoder.summary()
  return decoder



class VAE_movie(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE_movie, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.m_vae_loss_tracker = keras.metrics.Mean(name='m_vae_loss')
        self.a_vae_loss_tracker = keras.metrics.Mean(name='a_vae_loss')
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.kl_loss_tracker, self.reconstruction_loss_tracker,
                self.m_vae_loss_tracker, self.a_vae_loss_tracker]

    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
 
            outputs_mu, outputs_log_var = self.decoder(z)
            #print(data.shape, outputs_mu.shape)    ####デバッグ
            #sys.exit(1)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, outputs_mu)
            )
            reconstruction_loss *= 28 * 28 

            delta = 1e-7
            outputs_sigma2 = tf.math.exp(outputs_log_var)
            m_vae_loss = (data - outputs_mu)**2 / (outputs_sigma2+delta)
            m_vae_loss = 0.5 * tf.reduce_mean(m_vae_loss)

            a_vae_loss = tf.math.log(2 * 3.14 * outputs_sigma2)
            a_vae_loss = 0.5 * tf.reduce_mean(a_vae_loss)

            z_sigma2 = tf.math.exp(z_log_var)
            kl_loss = 1 + z_log_var - z_mean**2 - z_sigma2
            kl_loss =  -0.5 * tf.reduce_mean(kl_loss)

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.m_vae_loss_tracker.update_state(m_vae_loss)
        self.a_vae_loss_tracker.update_state(a_vae_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "a_vae_loss" : self.a_vae_loss_tracker.result(),
            "m_vae_loss" : self.m_vae_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss": self.total_loss_tracker.result(), 
        }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        outputs_mu, outputs_log_var = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, outputs_mu)
        )
        reconstruction_loss *= 28 * 28

        delta = 1e-7

        outputs_sigma2 = tf.math.exp(outputs_log_var)
        m_vae_loss = (data - outputs_mu)**2 / (outputs_sigma2+delta)
        m_vae_loss = 0.5 * tf.reduce_mean(m_vae_loss)
        m_vae_loss *= 28*28

        a_vae_loss = tf.math.log(2 * 3.14 * outputs_sigma2)
        a_vae_loss = 0.5 * tf.reduce_mean(a_vae_loss)

        z_sigma2 = tf.math.exp(z_log_var)
        kl_loss = 1 + z_log_var - z_mean**2 - z_sigma2
        kl_loss =  -0.5 * tf.reduce_mean(kl_loss)

        total_loss = reconstruction_loss + kl_loss

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.m_vae_loss_tracker.update_state(m_vae_loss)
        self.a_vae_loss_tracker.update_state(a_vae_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "m_vae_loss" : self.m_vae_loss_tracker.result(),
            "a_vae_loss" : self.a_vae_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss": self.total_loss_tracker.result(), 
        }

    def call(self, input):
        z_mean, z_log_var, z = self.encoder(input)
        reconstruction = self.decoder(z)
        return reconstruction

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    def get_config(self):
        return {"encoder": self.encoder, "decoder": self.decoder}

    def save(self, encoderfile, decoderfile):
        self.encoder.save(encoderfile)
        self.decoder.save(decoderfile)

    def load(self, encoderfile, decoderfile):
        self.encoder = keras.models.load_model(encoderfile, custom_objects={'Sampling': Sampling()})
        self.decoder = keras.models.load_model(decoderfile)        

    def img_predict(self, img):
        z_mean, z_log_var, z = self.encoder.predict(img)
        output_mu, output_sigma = self.decoder.predict(z)
        return output_mu, output_sigma

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
