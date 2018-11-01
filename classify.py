import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import load_model
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO
import pandas as pd
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="Path to trained model")
ap.add_argument("-l", "--labelbin", required=True,
                help="Path to label binarizer")
ap.add_argument("-i", "--file", required=True,
                help="Path to input sound file")
args = vars(ap.parse_args())

FRAME_SIZE = 5.e-2  # msecs

print("[INFO] Loading sound file")
[Fs, x] = audioBasicIO.readAudioFile(args['file'])
x = audioBasicIO.stereo2mono(x)
features, _ = audioFeatureExtraction.stFeatureExtraction(
    x, Fs, FRAME_SIZE * Fs, FRAME_SIZE / 2 * Fs)

inputArray = np.expand_dims(features, axis=3)

print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

first_layer = model.get_layer(index=0)
required_input_shape = first_layer.get_config()['batch_input_shape'][1:]
print('[INFO] Required Shape:', required_input_shape)
print('[INFO] Actual shape:', inputArray.shape)
# Adjust input to match required shape
if required_input_shape[1] > inputArray.shape[1]:
    zerosArray = np.zeros((1, required_input_shape[1] - inputArray.shape[1], 1), dtype=inputArray.dtype)
    inputArray = np.concatenate( (inputArray, zerosArray), axis = 1)
else:
    inputArray = inputArray[:, :required_input_shape[1], :]

print('[INFO] Post processed actual shape:', inputArray.shape)
print("[INFO] classifying sound...")
proba = model.predict(np.expand_dims(inputArray, axis=0))[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

label_with_predictions = {}
for i in range(len(proba)):
    label_with_predictions[lb.classes_[i]] = proba[i]

print("[INFO] Probabilities:", label_with_predictions)

print("[INFO] Prediction {}".format(label))
