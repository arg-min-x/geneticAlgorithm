
import numpy as np
from GA import *


genes_table = []
for ii in range(5000):
    gene = networkGenesMnist()
    gene.getRandomGenes()
    gene = gene.mutateGene(gene, hash_table=genes_table)
    genes_table.append(gene.hashGene())
# ga = genticAlgorithm((28, 28), 10, networkGenesMnist, mnistGenetic)
# fitness = ga.main(pop_num=10)
