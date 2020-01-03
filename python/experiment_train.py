#!/usr/bin/env python3

from data import loaders
from models.autoencoder_128_128_3 import model_fn
from models.train import train

params = {}
params['data_dir'] = '.../Geometry_Images/LSS_airplane'  # Path to data
params['model_dir'] = './log/LSS_airplane'  # Path to save model
params['parameterization_suffix'] = 'arcSmi'
params['rgb_size'] = 128
params['rgb_channels'] = 3
params['gi_size'] = 128
params['gi_channels'] = 3
params['use_mask'] = True
params['loadFiles'] = False
params['generateMeanShape'] = True
params['SelectedViews'] = ['view000', 'view001', 'view002', 'view010', 'view011', 'view012',
                           'view020', 'view021', 'view022', 'view030', 'view031', 'view032',
                           'view040', 'view041', 'view042', 'view050', 'view051', 'view052',
                           'view060', 'view061', 'view062', 'view070', 'view071', 'view072',
                           'view200', 'view201', 'view202', 'view210', 'view211', 'view212',
                           'view220', 'view221', 'view222', 'view230', 'view231', 'view232',
                           'view240', 'view241', 'view242', 'view250', 'view251', 'view252',
                           'view260', 'view261', 'view262', 'view270', 'view271', 'view272',
                           'view300', 'view301', 'view302', 'view310', 'view311', 'view312',
                           'view320', 'view321', 'view322', 'view330', 'view331', 'view332',
                           'view340', 'view341', 'view342', 'view350', 'view351', 'view352',
                           'view360', 'view361', 'view362', 'view370', 'view371', 'view372',
                           'view400', 'view401', 'view402', 'view410', 'view411', 'view412',
                           'view420', 'view421', 'view422', 'view430', 'view431', 'view432',
                           'view440', 'view441', 'view442', 'view450', 'view451', 'view452',
                           'view460', 'view461', 'view462', 'view470', 'view471', 'view472']
# model specification
params['lr'] = 0.0001  # lr = 0.0001 for airplane and 0.00001 for car
params['batch_size'] = 64
params['epochs'] = 50
params['epochs_between_evals'] = 2
params['instances_between_checkpoints'] = 2000

loaders.save_params(params)
train(model_fn, params)
