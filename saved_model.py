import tensorflow as tf
import tensorflow_io as tfio

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if physical_devices:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)

# VAEモデル読み込み
VAEModulePath = '/content/VAE_movie'
import sys
sys.path.append(VAEModulePath)
import VAE_movie
vae = VAE_movie.VAE_movie(None, None)
vae.load('models_movie/encoder.h5','models_movie/decoder.h5')
print("after load VAE")

# VAEモデルの入力を設定(saved model用)
x = tf.TensorSpec((None,6,64,64,3), tf.float32, name='x')   #元々は((None,6,128,128,3)
concrete_function = tf.function(lambda x: vae(x)).get_concrete_function(x)
vae.summary()
print("after input settiing ")
# savedModelとして保存
import os
module_no_signatures_path = os.path.join("tmpdir", 'vae_movie/1/')
tf.saved_model.save(vae, module_no_signatures_path)
