import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.utils import np_utils
from keras.optimizers import Adam

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from models.conv2d import Conv2DNN
from sys import stdout

import argparse
import pickle
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of sound files)")
ap.add_argument("-m", "--model", default="nn.model",
                help="path to output model")
ap.add_argument("-l", "--labelbin", default="lb.pickle",
                help="path to output label binarizer")
args = vars(ap.parse_args())

FRAME_SIZE = 5.e-2 #msecs
KERNEL_SIZE = 12
BATCH_SIZE = 32
EPOCHS = 750
LR = 5.e-6
DATASET = args['dataset']
OPTIMIZER = 'ADAM'

SOUND_PICKLE = 'sound_data.npy'
EMOTION_LABEL_PICKLE = 'emotion_labels.npy'

fileList = os.listdir(args['dataset'])

if (not os.path.isfile(SOUND_PICKLE)) or (not os.path.isfile(EMOTION_LABEL_PICKLE)):
    feeling_list=[]
    for item in fileList:
        if item[-3:] != 'wav':
            print('[DEBUG] Error!', item)
        print('Labeling:', item)
        if item[6:8]=='03' and int(item[18:20])%2==0 or (item[:3]=='gio' and item[4]=='f'):
            feeling_list.append('female_happy')
        elif (item[6:8]=='03' and int(item[18:20])%2==1) or item[:1]=='h' or (item[:3]=='gio' and item[4]=='m'):
            feeling_list.append('male_happy')
        elif item[6:8]=='04' and int(item[18:20])%2==0 or (item[:3]=='tri' and item[4]=='f'):
            feeling_list.append('female_sad')
        elif (item[6:8]=='04' and int(item[18:20])%2==1) or item[:2]=='sa' or (item[:3]=='tri' and item[4]=='m'):
            feeling_list.append('male_sad')
        elif (item[6:8]=='05' and int(item[18:20])%2==0) or (item[:3]=='rab' and item[4]=='f'):
            feeling_list.append('female_angry')
        elif (item[6:8]=='05' and int(item[18:20])%2==1) or  item[:1]=='a' or (item[:3]=='rab' and item[4]=='m'):
            feeling_list.append('male_angry')
        elif (item[6:8]=='06' and int(item[18:20])%2==0) or (item[:3]=='pau' and item[4]=='f'):
            feeling_list.append('female_fearful')
        elif (item[6:8]=='06' and int(item[18:20])%2==1) or item[:1]=='f' or (item[:3]=='pau' and item[4]=='m'):
            feeling_list.append('male_fearful')
        elif ((item[6:8]=='02' or item[6:8]=='01') and int(item[18:20])%2==1) or item[:1]=='n' or (item[:3]=='neu' and item[4]=='m'):
            feeling_list.append('male_neutral')
        elif ((item[6:8]=='02' or item[6:8]=='01') and int(item[18:20])%2==0) or (item[:3]=='neu' and item[4]=='f'):
            feeling_list.append('female_neutral')
        elif (item[6:8]=='08' and int(item[18:20])%2==1) or item[:2] == 'su' or (item[:3]=='sor' and item[4]=='m'):
            feeling_list.append('male_surprised')
        elif (item[6:8]=='08' and int(item[18:20])%2==0) or (item[:3]=='sor' and item[4]=='f'):
            feeling_list.append('female_surprised')
        elif item[:1] == 'd' or (item[:3]=='dis' and item[4]=='m') or (item[6:8]=='07' and int(item[18:20])%2==1):
            feeling_list.append('male_disgust')
        elif (item[:3]=='dis' and item[4]=='f') or (item[6:8]=='07' and int(item[18:20])%2==0):
            feeling_list.append('female_disgust')
        stdout.write('\033[F')
        
    labels = np.array(feeling_list)
    np.save(EMOTION_LABEL_PICKLE, labels)
    bookmark=0
    for file_name in fileList:
        print('[INFO] Extracting features - {} / {} - {}'.format(bookmark + 1, len(fileList), file_name))
        [Fs, x] = audioBasicIO.readAudioFile(args['dataset'] + '/' + file_name)
        x = audioBasicIO.stereo2mono(x)
        features, feature_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, FRAME_SIZE * Fs, FRAME_SIZE / 2 * Fs);
        # npArray = np.array([np.array(feature, copy=False) for feature in features], copy=False, ndmin=3)
        npArray = np.array(features, ndmin=3)
        if (bookmark == 0):
            # soundData = pd.DataFrame(data=npArray, columns=feature_names)
            soundData = npArray
            columns = np.array(feature_names)
        else:
            if soundData.shape[2] > npArray.shape[2]:
                npArray = np.pad(npArray, ( (0, 0), (0, 0), (0, soundData.shape[2] - npArray.shape[2]) ), 'constant')
            else:
                soundData = np.pad(soundData, ( (0, 0), (0, 0), (0, npArray.shape[2] - soundData.shape[2]) ), 'constant')
            soundData = np.concatenate((soundData, npArray))
        bookmark=bookmark+1
        stdout.write('\033[F')
    np.save(SOUND_PICKLE, soundData)
else:
    print('[INFO] Using pickled data')
    soundData = np.load(SOUND_PICKLE)
    labels = np.load(EMOTION_LABEL_PICKLE)

shuffled = shuffle(soundData)

lb = LabelBinarizer()
encodedLabels = lb.fit_transform(labels)
print('[INFO] Unique labels:', len(lb.classes_))

(trainX, testX, trainY, testY) = train_test_split(soundData, encodedLabels, test_size=0.2, random_state=61)

trainX = np.expand_dims(trainX, axis = 3)
testX = np.expand_dims(testX, axis = 3)
print('[INFO] TrainX Data Shape:', trainX.shape)
print('[INFO] TrainY Data shape:', trainY.shape)
model_shape = (soundData.shape[1], soundData.shape[2], 1)

print("[INFO] Setting up model...")
model = Conv2DNN.build(model_shape, len(lb.classes_))

CONV_COUNT, DROPOUT_COUNT = Conv2DNN.countLayers(model)

opt = keras.optimizers.rmsprop(lr=LR, decay=LR / EPOCHS) if OPTIMIZER == 'RMS' else Adam(lr=LR, decay=LR / EPOCHS)

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testX, testY))

#save model
print("[INFO] Saving model")
model.save(args['model'])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

CONV_COUNT, DROPOUT_COUNT = Conv2DNN.countLayers(model)

plot_file_name = 'r{0}_l{1:.1e}_b{2}_k{3}_conv2d{4}-{5}_{6}_{7}_{8:.1e}fs_aug.png'.format(EPOCHS, LR, BATCH_SIZE, KERNEL_SIZE, CONV_COUNT, DROPOUT_COUNT, OPTIMIZER, DATASET, FRAME_SIZE)

# Loss History
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('plots/loss_' + plot_file_name)

# Accuracy history
plt.plot(cnnhistory.history['acc'])
plt.plot(cnnhistory.history['val_acc'])
plt.title('model loss & accuracy')
plt.ylabel('accuracy / loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss', 'train_acc', 'test_acc'], loc='upper left')
plt.savefig('plots/' + plot_file_name)

print('[INFO] Predicting test data...')
preds = model.predict(testX, 
                         batch_size=32, 
                         verbose=1)
preds = preds.argmax()
preds = preds.astype(int).flatten()

labelPreds = (lb.inverse_transform(preds))

actual = y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform(actual))

finalDf = pd.DataFrame({'predicted_values': labelPreds, 'actual_values': actual})

print(finalDf.groupby('actual_values').count())
print(finalDf.groupby('predicted_values').count())

