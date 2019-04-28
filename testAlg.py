#import keras
from keras.datasets       import mnist
from keras.models         import Model
from keras.layers         import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras                import backend as K
import numpy as np
import logging
from GA import networkGenes, getNetwork, testFitness

def get_mnist_mlp():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes  = 10 #dataset dependent
    batch_size  = 64
    epochs      = 4
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

(nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs) = get_mnist_mlp()


filters = [8, 8, 8]
kernel_size = 2
output_size = 10
dense_size = 32
genes = networkGenes(input_shape=(28, 28, 1),
                     num_layers=3,
                     filters=filters,
                     kernel_size=kernel_size,
                     output_size=10,
                     dense_size=32)
inputs, outputs = getNetwork(genes)

accuracy = testFitness(inputs, outputs, get_mnist_mlp)
# model = Model(inputs, outputs)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train,
#           y_train,
#           validation_data=(x_test, y_test))  # starts training