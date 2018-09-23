from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class Conv2DNN:
    @staticmethod
    def build(shape, classes):
        model = Sequential()

        model.add(Conv2D(8, (3, 3), padding='same', input_shape=shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.25))

        model.add(Conv2D(12, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.25))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

def getLastLayerInputShape(model):
    last_layer = model.get_layer(index=-1)
    config = last_layer.get_config()
    try:
        print('Layer: {}, has shape: {}'.format(config['name'], config['units']))
    except:
        print('Layer: {} has no units'.format(config['name']))
