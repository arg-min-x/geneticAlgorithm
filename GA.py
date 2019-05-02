#import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils.np_utils import to_categorical
import numpy as np
import random


class genticAlgorithm():
    def __init__(self, input_shape, num_classes, gene_class, network_class):
        self.input_shape = input_shape  # shape of the input layer to the netowork
        self.num_classes = num_classes  # number of classes to predict
        self.pop = []  # The current population of genes
        self.gene_class = gene_class
        self.network_class = network_class
        self.fitness = []
    def get_pop_fitness(self):
        """"
        Get the fitness of the population
        """
        for genes, ii in zip(self.pop, range(len(self.pop))):
            if self.fitness[ii] < 0:
                network = self.network_class()
                inputs, outputs = network.getNetwork(genes)
                self.fitness[ii] = network.testFitness(inputs, outputs)

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

    def main(self, pop_num=5):
        """

        :return:
        """
        num_kill = 1

        # Get random genes to fill out the population
        for ii in range(pop_num):
            genes = self.gene_class()
            genes.getRandomGenes()
            self.pop.append(genes)

        # Initialize fitness
        self.fitness = [-1 for ii in range(pop_num)]

        # main loop
        for ii in range(4):
            # get the fitness of the population
            self.get_pop_fitness()

            # sort fitness and genes
            sort_inds = list(np.argsort(-np.array(self.fitness)))
            self.fitness = [self.fitness[ii] for ii in sort_inds]
            self.pop = [self.pop[ii] for ii in sort_inds]
            print(self.fitness)

            # delete the lowest fitness genes and
            self.fitness[-1] = -1
            self.pop[-1].mutateGene()


class networkGenesMnist:
    def __init__(self):
        self.input_shape = (28, 28, 1)
        self.output_size = 10
        self.dense_size = 32
        self.m_para = {'filters': [],
                       'kernel_size': [],
                       'num_layers': []}
        self.m_para_r = {'filters': (4, 9),
                         'kernel_size': (1, 3),
                         'num_layers': (1, 3)}

    def getRandomGenes(self):
        """
        :return:
        """
        self.m_para["num_layers"] = random.randint(*self.m_para_r["num_layers"])
        for ii in range(self.m_para_r["num_layers"][1]):
            self.m_para["filters"].append(
                random.randint(*self.m_para_r["filters"]))
        self.m_para["kernel_size"] = random.randint(
            *self.m_para_r["kernel_size"])

    def mutateGene(self):
        """

        :return:
        """
        # randomly chooses a gene to mutate
        mut_gene = random.choice(list(self.m_para.keys()))
        print(mut_gene)
        print(self.m_para[mut_gene])
        if mut_gene == 'filters':
            self.m_para[mut_gene] = []
            print(range(self.m_para_r["num_layers"][1]))
            for ii in range(self.m_para_r["num_layers"][1]):
                self.m_para[mut_gene].append(
                    random.randint(*self.m_para_r[mut_gene]))
        else:
            self.m_para[mut_gene] = random.randint(*self.m_para_r[mut_gene])
        print(self.m_para[mut_gene])
        return 0

    def crossOverGenes(self, mom, dad):
        """

        :param mom:
        :param dad:
        :return:
        """

class mnistGenetic:
    def __init__(self):
        """

        """

    def getData(self):
        """Retrieve the MNIST dataset and process the data."""
        # Set defaults.
        nb_classes = 10  # dataset dependent
        batch_size = 64
        epochs = 4
        input_shape = (784,)

        # Get the data.
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        x_train = np.reshape(x_train, x_train.shape + (1,))
        x_test = np.reshape(x_test, x_test.shape + (1,))

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

    def getNetwork(self, genes):
        """

        :param genes:
        :return:
        """
        inputs = Input(shape=genes.input_shape)
        for ii in range(genes.m_para["num_layers"]):
            if ii == 0:
                x = Conv2D(filters=genes.m_para["filters"][ii],
                           kernel_size=genes.m_para["kernel_size"])(inputs)
            else:
                x = Conv2D(filters=genes.m_para["filters"][ii],
                           kernel_size=genes.m_para["kernel_size"])(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            x = MaxPooling2D()(x)

        x = Flatten()(x)
        x = Dense(genes.dense_size, activation='sigmoid')(x)
        outputs = Dense(genes.output_size, activation='softmax')(x)
        return inputs, outputs

    def testFitness(self, inputs, outputs):
        """
        :param inputs:
        :param outputs:
        :return:
        """
        (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs) = self.getData()

        model = Model(inputs, outputs)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train,
                  y_train,
                  validation_data=(x_test, y_test),
                  verbose=0)  # starts training

        return model.history.history['val_acc'][0]
