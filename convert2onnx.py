#!/usr/bin/env python

# ./convert2onnx.py denoising5_tf2-93.h5 CDCHWfDenoising_356990_20201123_0.onnx

import os, sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import onnxmltools
#import keras2onnx

print ('Python version: ' + str(sys.version_info))
print ('TensorFlow version: ' + str(tf.__version__))
print ('ONNXMLTools version:' + str(onnxmltools.__version__))
#print ('Keras2Onnx version:' + str(keras2onnx.__version__))

# Edit file names
input_keras_model = 'denoising5_tf2.h5'
output_onnx_model = 'CDCHWfDenoising_356990_20201123_2.onnx'

if len(sys.argv) > 1:
    input_keras_model = sys.argv[1]
if len(sys.argv) > 2:    
    output_onnx_model = sys.argv[2]

# Load your Keras model
keras_model = load_model(input_keras_model)

# Convert the Keras model into ONNX
onnx_model = onnxmltools.convert_keras(keras_model)
#onnx_model = keras2onnx.convert_keras(keras_model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)
#keras2onnx.utils.save_model(onnx_model, output_onnx_model)
