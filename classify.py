from keras.models import load_model
import librosa
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

print("[INFO] Loading sound file")
sound = pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load(args['file'], res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                    sr=sample_rate, 
                                    n_mfcc=13),
                axis=0)
feature = mfccs

sound.loc[0] = [feature]
flatDf = pd.DataFrame(sound['feature'].values.tolist())
inputArray = np.expand_dims(flatDf, axis=2)

print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

first_layer = model.get_layer(index=0)
required_input_shape = first_layer.get_config()['batch_input_shape'][1:]

# Adjust input to match required shape
if required_input_shape[0] > inputArray.shape[1]:
    zerosArray = np.zeros((1, required_input_shape[0] - inputArray.shape[1], 1), dtype=inputArray.dtype)
    inputArray = np.concatenate( (inputArray, zerosArray), axis = 1)
else:
    inputArray = inputArray[:required_input_shape[0]]


print("[INFO] classifying sound...")
proba = model.predict(inputArray)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

label_with_predictions = {}
for i in range(len(proba)):
    label_with_predictions[lb.classes_[i]] = proba[i]

print("[INFO] Probabilities:", label_with_predictions)

print("[INFO] Prediction {}".format(label))
