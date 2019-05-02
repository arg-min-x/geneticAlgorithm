#import keras
from keras.datasets       import mnist
from keras.models         import Model
from keras.layers         import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras                import backend as K
import numpy as np
import logging
from GA import *

ga = genticAlgorithm((28, 28), 10, networkGenesMnist, mnistGenetic)
fitness = ga.main(pop_num=3)



# filters = [8, 8, 8]
# kernel_size = 2
# output_size = 10
# dense_size = 32
# genes = networkGenes(input_shape=(28, 28, 1),
#                      num_layers=3,
#                      filters=filters,
#                      kernel_size=kernel_size,
#                      output_size=10,
#                      dense_size=32)
# genes = networkGenesMnist()
# genes.getRandomGenes()
# genes.mutateGene()
# print(genes.m_para['filters'])
# mnist_class = mnistGenetic()
# inputs, outputs = mnist_class.getNetwork(genes)
# mnist_class.testFitness(inputs, outputs)
# inputs, outputs = getNetwork(genes)
#
# accuracy = testFitness(inputs, outputs, get_mnist_mlp)
# model = Model(inputs, outputs)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train,
#           y_train,
#           validation_data=(x_test, y_test))  # starts training