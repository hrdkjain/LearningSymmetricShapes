import tensorflow.compat.v1 as tf
import os
import sys
sys.path.append('..')
from models.autoencoder_128_128_3 import model_fn

# this script is tested on tensorflow version 2.2 and may not work on older/newer versions

tf.disable_v2_behavior()

params = {}
params['model_dir'] = '../savedModel/LSS_airplane'
params['checkpoint'] = '35424'

# params['model_dir'] = '../savedModel/LSS_car'
# params['checkpoint'] = '14850'
params['pb_model_dir'] = os.path.join(params['model_dir'],'pb')

checkpoint_path = os.path.join(params['model_dir'], 'model.ckpt-' + params['checkpoint'])

network = tf.estimator.Estimator(
    model_dir=params['model_dir'],
    model_fn=model_fn,
    params=params,
)

def serving_input_receiver_fn():
    features = {'encoder_input': tf.placeholder(
        shape=[1, 128, 128, 3],
        dtype=float)
    }
    return tf.estimator.export.ServingInputReceiver(features, features)

network.export_saved_model(params['pb_model_dir'], serving_input_receiver_fn, checkpoint_path=checkpoint_path)
