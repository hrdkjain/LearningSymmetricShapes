import tensorflow as tf
from pathlib import Path
import os

# this script is tested on tensorflow version 2.2 and may not work on older/newer versions

saved_model_dir = os.path.join('.', 'pbLSSCarModel2', '1599348109')
saved_model_dir = ".\carModel"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.allow_custom_ops = True
tflite_model = converter.convert()

Path("tfliteModels").mkdir(exist_ok=True)
outputFile = os.path.join('tfliteModels', 'converted_car_model.tflite')
open(outputFile, "wb").write(tflite_model)