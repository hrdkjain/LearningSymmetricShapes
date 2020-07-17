#!/usr/bin/env python3

import os
from data import utils
from models.autoencoder_128_128_3 import model_fn
from models.predict import predict
import numpy as np

params = {}
params['model_dir'] = './savedModel/LSS_airplane'
params['checkpoint'] = '35424'
params['rgb_size'] = 128
params['gi_size'] = 128

checkpointPath = os.path.join(params['model_dir'], 'model.ckpt-' + params['checkpoint'])
rgb_imgs = ['tst_rgb/airplane_2.png',
            'tst_rgb/airplane_4.png',
            'tst_rgb/airplane_5.png',
            'tst_rgb/airplane_7.png']
params['batch_size'] = len(rgb_imgs)

# Compute output of the model by specifying input rgb image
prediction = predict(model_fn, params, rgb_imgs, checkpointPath)

# create a reflector to obtain the second half of the mesh
reflector = np.array([1,1,-1])

# extract the predictions and save as off file in the same path as the input rgb images
for i in range(len(rgb_imgs)):
    out = next(prediction)
    if isinstance(out, dict):
        slice1 = np.reshape(out['prediction'],[params['gi_size']*params['gi_size'],3])
        slice2 = slice1 * reflector
        utils.writeOff(rgb_imgs[i].replace('.png','.off'), np.concatenate((slice1,slice2),axis=0), params['gi_size'], False)
