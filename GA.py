#import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils.np_utils import to_categorical
import numpy as np
import random
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(2)
seed(1)

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

        num_mutate = 5

        # Get random genes to fill out the population
        for ii in range(pop_num):
            genes = self.gene_class()
            genes.getRandomGenes()
            self.pop.append(genes)

        # Initialize fitness
        self.fitness = [-1 for _ in range(pop_num)]

        # main loop
        for ii in range(5):
            # get the fitness of the population
            self.get_pop_fitness()

            # sort fitness and genes
            sort_inds = list(np.argsort(-np.array(self.fitness)))
            self.fitness = [self.fitness[ii] for ii in sort_inds]
            self.pop = [self.pop[ii] for ii in sort_inds]
            print(self.fitness)

            # delete half  of the genes
            del_genes = round(pop_num/2)

            # flag fitness for retraining
            for ii in range(del_genes, pop_num):
                self.fitness[ii] = -1

            breed_genes = self.pop[:del_genes]
            del_genes_ind = list(range(del_genes, pop_num))
            while breed_genes:
                if len(breed_genes) == 1:
                    self.pop[del_genes_ind[0]] = self.pop[0].mutateGene(breed_genes[0])
                    breed_genes.pop(0)
                    del_genes_ind.pop(0)
                else:
                    breed_genes_ind = random.choices(list(range(len(breed_genes))), k=2)
                    brother, sister = crossOverGenes(self.pop[breed_genes_ind[0]], self.pop[breed_genes_ind[1]])
                    self.pop[del_genes_ind[0]] = brother
                    self.pop[del_genes_ind[1]] = sister

                    breed_genes_ind.sort()
                    breed_genes.pop(breed_genes_ind[0])
                    breed_genes.pop(breed_genes_ind[1]-1)

                    del_genes_ind.pop(0)
                    del_genes_ind.pop(0)

            # Mutate some genes
            mutate_inds = np.random.choice(range(1, pop_num), size=num_mutate, replace=False)
            for m_ind in mutate_inds:
                mut_mult = random.randint(1, 2)
                # mutate one gene
                if mut_mult == 1:
                    self.pop[m_ind] = self.pop[1].mutateGene(self.pop[m_ind])
                # mutate two genes
                else:
                    self.pop[m_ind] = self.pop[1].mutateGene(self.pop[m_ind])
                    self.pop[m_ind] = self.pop[1].mutateGene(self.pop[m_ind])


class networkGenesMnist:
    def __init__(self):
        self.input_shape = (28, 28, 1)
        self.output_size = 10
        self.m_para = {'filters': [],
                       'kernel_size': [],
                       'num_layers': [],
                       'dense_size': []}
        self.m_para_r = {'filters': (4, 16),
                         'kernel_size': (1, 6),
                         'num_layers': (1, 5),
                         'dense_size': (16, 32)}

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
        self.m_para["dense_size"] = random.randint(*self.m_para_r["dense_size"])

    def mutateGene(self, parent, hash_table=None):
        """

        :return:
        """
        keep_mutating = 1
        num_tries = 0
        while keep_mutating:
            child = parent
            # randomly chooses a gene to mutate
            mut_gene = random.choice(list(parent.m_para.keys()))
            if mut_gene == 'filters':
                child.m_para[mut_gene] = []
                for ii in range(parent.m_para_r["num_layers"][1]):
                    child.m_para[mut_gene].append(
                        random.randint(*parent.m_para_r[mut_gene]))
            else:
                child.m_para[mut_gene] = random.randint(*parent.m_para_r[mut_gene])

            # check if the gene was already tried
            if child.hashGene() not in hash_table:
                keep_mutating = 0
            else:
                num_tries += 1
                print("duplicate gene")

            # try 50 times, then give up
            if num_tries > 50:
                keep_mutating = 0

        return child

    def hashGene(self):
        """
        :return:
        """
        # get all the values in the genes as  a list
        gene_keys = list(self.m_para.keys())
        genes = []
        for keys in gene_keys:
            if isinstance(self.m_para[keys], list):
                for item in self.m_para[keys]:
                    genes.append(item)
            else:
                genes.append(self.m_para[keys])

        # convert to a string
        genes = [str(item) for item in genes]
        genes_str = ""
        for gene_s in genes:
            genes_str += gene_s
        return genes_str


def crossOverGenes(mom, dad):
    """

    :param mom:
    :param dad:
    :return:
    """
    brother = dad
    sister = mom

    # pick a random gene and cross over the parents
    cross_gene = random.choice(list(mom.m_para.keys()))
    mom_gene_tmp = mom.m_para[cross_gene]
    sister.m_para[cross_gene] = dad.m_para[cross_gene]
    brother.m_para[cross_gene] = mom_gene_tmp
    return brother, sister


class mnistGenetic:
    def __init__(self):
        """

        """

    def getData(self):
        """Retrieve the MNIST dataset and process the data."""
        # Set defaults.
        nb_classes = 10  # dataset dependent
        batch_size = 128
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
        # Set random seed
        set_random_seed(2)
        seed(1)

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
            # x = MaxPooling2D()(x)

        x = Flatten()(x)
        x = Dense(genes.m_para["dense_size"], activation='sigmoid')(x)
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
                  verbose=0,
                  epochs=2)  # starts training

        return model.history.history['val_acc'][0]
