import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import load_model
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO
import pandas as pd
import numpy as np
import pickle
import os

FRAME_SIZE = 5.e-2  # msecs

def classifyEmotion(filePath):
    modelPath = 'emotion/emo_nn.model'
    labelPath = 'emotion/lb.pickle'
    print("[INFO] Loading sound file")
    [Fs, x] = audioBasicIO.readAudioFile(filePath)
    x = audioBasicIO.stereo2mono(x)
    features, _ = audioFeatureExtraction.stFeatureExtraction(
        x, Fs, FRAME_SIZE * Fs, FRAME_SIZE / 2 * Fs)
    inputArray = np.expand_dims(features, axis=3)

    print("[INFO] loading network...")
    model = load_model(modelPath)
    lb = pickle.loads(open(labelPath, "rb").read())

    first_layer = model.get_layer(index=0)
    required_input_shape = first_layer.get_config()['batch_input_shape'][1:]

    # Adjust input to match required shape
    if required_input_shape[1] > inputArray.shape[1]:
        zerosArray = np.zeros((1, required_input_shape[1] - inputArray.shape[1], 1), dtype=inputArray.dtype)
        inputArray = np.concatenate( (inputArray, zerosArray), axis = 1)
    else:
        inputArray = inputArray[:, :required_input_shape[1], :]


    print("[INFO] classifying sound...")
    proba = model.predict(np.expand_dims(inputArray, axis=0))[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]

    label_with_predictions = {}
    for i in range(len(proba)):
        label_with_predictions[lb.classes_[i]] = proba[i]

    print("[INFO] Probabilities:", label_with_predictions)

    print("[INFO] Prediction {}".format(label))
    return label
