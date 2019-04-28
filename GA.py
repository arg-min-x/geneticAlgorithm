#import keras
from keras.datasets       import mnist
from keras.models         import Model
from keras.layers         import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation

import numpy as np

class genticAlgorithm():
    def __init__(self):
        self.input_shape = input_shape  # shape of the input layer to the netowork
        self.num_classes = num_classes  # number of classes to predict
        self.pop = []  # The current population of genes

    def get_pop_fitness(self):
        """"
        Get the fitness of the population
        """

    def select_mating_pool(self):
        """
        return the mating pool
        :return:
        """

    def cross_over(self):
        """
        cross over the mating pool to create a new generation
        :return:
        """

    def mutate(self):
        """
        Mutate the genes in the new generation
        :return:
        """

class networkGenes:
    def __init__(self,input_shape, num_layers, filters, kernel_size, output_size, dense_size):
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.dense_size = dense_size


def getNetwork(genes):

    inputs = Input(shape=genes.input_shape)
    for ii in range(genes.num_layers):
        if ii == 1:
            x = Conv2D(filters=genes.filters[ii], kernel_size=genes.kernel_size)(inputs)
        else:
            x = Conv2D(filters=genes.filters[ii], kernel_size=genes.kernel_size)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(genes.dense_size, activation='sigmoid')(x)
    outputs = Dense(genes.output_size, activation='softmax')(x)
    return inputs, outputs


def getRandomGenes()


def testFitness(inputs, outputs, getData):
    (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs) = getData()

    model = Model(inputs, outputs)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test))  # starts training

    return model.history.history['val_acc']