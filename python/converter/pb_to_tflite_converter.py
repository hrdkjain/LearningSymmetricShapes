import tensorflow as tf
from pathlib import Path
import os

# this script is tested on tensorflow version 2.2 and may not work on older/newer versions
params = {}
params['model_dir'] = '../savedModel/LSS_airplane'
# params['model_dir'] = '../savedModel/LSS_car'
params['pb_model_dir'] = os.path.join(params['model_dir'],'pb','1599650016')
params['tflite_model_dir'] = os.path.join(params['model_dir'],'tflite')

converter = tf.lite.TFLiteConverter.from_saved_model(params['pb_model_dir'])
converter.allow_custom_ops = True
tflite_model = converter.convert()

Path(params['tflite_model_dir']).mkdir(exist_ok=True)
outputFile = os.path.join(params['tflite_model_dir'], 'model.tflite')
open(outputFile, "wb").write(tflite_model)