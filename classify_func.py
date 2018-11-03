
from keras.models import load_model
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO
import pandas as pd
import numpy as np
import pickle

FRAME_SIZE = 5.e-2  # msecs

modelPath = 'emotion/emo_nn.model'
labelPath = 'emotion/lb.pickle'
print("[INFO] loading network...")
lb = pickle.loads(open(labelPath, "rb").read())
model = load_model(modelPath)

first_layer = model.get_layer(index=0)
model._make_predict_function()
print('[INFO] Emotion label model loaded')

def classifyEmotion(filePath):
    print("[INFO] Loading sound file")
    [Fs, x] = audioBasicIO.readAudioFile(filePath)
    x = audioBasicIO.stereo2mono(x)
    features, _ = audioFeatureExtraction.stFeatureExtraction(
        x, Fs, FRAME_SIZE * Fs, FRAME_SIZE / 2 * Fs)
    inputArray = np.expand_dims(features, axis=3)


    first_layer = model.get_layer(index=0)
    required_input_shape = first_layer.get_config()['batch_input_shape'][1:]

    # Adjust input to match required shape
    if required_input_shape[1] > inputArray.shape[1]:
        zerosArray = np.zeros((required_input_shape[0], required_input_shape[1] - inputArray.shape[1], 1), dtype=inputArray.dtype)
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
