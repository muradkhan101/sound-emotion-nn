import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
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
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from models.conv1d import Conv1DNN

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

KERNEL_SIZE = 12
BATCH_SIZE = 32
EPOCHS = 500
LR = 1.e-6
DATASET = args['dataset']
OPTIMIZER = 'ADAM'

SOUND_PICKLE = 'sound_data.pickle'
EMOTION_LABEL_PICKLE = 'emotion_labels.pickle'

fileList = os.listdir(args['dataset'])

if (not os.path.isfile(SOUND_PICKLE)) or (not os.path.isfile(EMOTION_LABEL_PICKLE)):
    feeling_list=[]
    for item in fileList:
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
        
    labels = pd.DataFrame(feeling_list)
    labels.to_pickle(EMOTION_LABEL_PICKLE)

    soundData = pd.DataFrame(columns=['feature'])
    bookmark=0
    for index,y in enumerate(fileList):
        X, sample_rate = librosa.load(args['dataset'] + '/' +y, res_type='kaiser_fast',sr=22050*2, duration=3.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13),
                        axis=0)
        feature = mfccs
        normalized = normalize([feature], norm='l1')
        soundData.loc[bookmark] = [normalized[0]]
        bookmark=bookmark+1
    soundData.to_pickle(SOUND_PICKLE)

else:
    soundData = pd.read_pickle(SOUND_PICKLE)
    labels = pd.read_pickle(EMOTION_LABEL_PICKLE)

df3 = pd.DataFrame(soundData['feature'].values.tolist())
newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})

rnewdf = shuffle(newdf)
rnewdf=rnewdf.fillna(0)
maskingDf = np.random.rand(len(rnewdf)) < 0.8

train = rnewdf[maskingDf]
test = rnewdf[~maskingDf]

trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

model_shape = X_train[0].shape

print("[INFO] Setting up model...")
print('Model shape', model_shape)
model = Conv1DNN.build(model_shape[0], 14, KERNEL_SIZE)

opt = keras.optimizers.rmsprop(lr=LR, decay=LR / EPOCHS) if OPTIMIZER == 'RMS' else Adam(lr=LR, decay=LR / EPOCHS)

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_testcnn, y_test))

#save model
print("[INFO] Saving model")
model.save(args['model'])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

plot_file_name = 'r{0}_l{1:.1f}_b{2}_k{3}_conv{4}_{5}_{}.png'.format(EPOCHS, LR, BATCH_SIZE, KERNEL_SIZE, 4, OPTIMIZER, DATASET)

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
preds = model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)
preds = preds.argmax(axis=1)
preds = preds.astype(int).flatten()

labelPreds = (lb.inverse_transform(preds))

actual = y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform(actual))

finalDf = pd.DataFrame({'predicted_values': labelPreds, 'actual_values': actual})

print(finalDf.groupby('actual_values').count())
print(finalDf.groupby('predicted_values').count())

