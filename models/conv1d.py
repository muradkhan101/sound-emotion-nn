from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense

class Conv1DNN:
    @staticmethod
    def build(input_shape, num_classes, kernel_size = 8):
        model = Sequential()
        model.add(Conv1D(input_shape, kernel_size, padding='same',
                        input_shape=(input_shape, 1)))
        model.add(Activation('relu'))

        model.add(Conv1D(input_shape, kernel_size, padding='same'))
        last_layer = model.get_layer(index=-1)
        config = last_layer.get_config()
        print('Layer: {} has no units'.format(config['name']))

        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # model.add(Dropout(0.1))
        model.add(MaxPooling1D(pool_size=(8)))

        model.add(Conv1D(input_shape, kernel_size, padding='same',))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1))

        model.add(Dropout(0.2))
        last_layer = model.get_layer(index=-1)
        config = last_layer.get_config()
        print('Layer: {} has no units'.format(config['name']))
        
        model.add(Conv1D(input_shape, kernel_size, padding='same',))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1))

        model.add(Flatten())
        print('NUm Classes', num_classes)
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        return model
