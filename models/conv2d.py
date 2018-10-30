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

        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Dropout(0.25))
        
        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Dropout(0.25))

        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(12, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.25))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(8, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))


        model.add(Conv2D(10, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.25))

        model.add(Conv2D(10, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(12, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.25))

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
    
    @staticmethod
    def countLayers(model):
        CONV_LAYERS = 0
        DROPOUT_LAYERS = 0
        currLayer = 0
        while True:
            try:
                layer = model.get_layer(index=currLayer)
                config = layer.get_config()
                name = config['name']
                if 'conv' in name:
                    CONV_LAYERS = CONV_LAYERS + 1
                elif 'dropout' in name:
                    DROPOUT_LAYERS = DROPOUT_LAYERS + 1
                currLayer += 1
            except Exception as err:
                print(err)
                return CONV_LAYERS, DROPOUT_LAYERS
            

def getLastLayerInputShape(model):
    last_layer = model.get_layer(index=-1)
    config = last_layer.get_config()
    try:
        print('Layer: {}, has shape: {}'.format(config['name'], config['units']))
    except:
        print('Layer: {} has no units'.format(config['name']))
