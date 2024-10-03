import numpy as np
import random


class AG_MOM():
    def __init__(self, N, G, mut_op, cross_op, T, xl, xu):
        self.N = N
        self.G = G
        self.mut_op = mut_op
        self.cross_op = cross_op
        self.T = T
        self.xl = xl
        self.xu = xu

    def create_vectors(self):
        vectors = []
        dst = 1/self.N
        while dst < 1:
            vector = [0+dst, 1-dst]
            vectors.append(vector)
            dst += dst
        return vectors

    def neighbors(self):
        N = self.N
        vectors = self.create_vectors()
        dsts = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                dsts[i, j] = np.linalg.norm(vectors[i] - vectors[j])

        neighbors = []

        for i in range(N):
            nearest_indexes = np.argsort(neighbors[i])[:self.T]
            neighbors.append(nearest_indexes)

        return neighbors

    def initialize_pop(self):
        pop = []
        for i in range(self.N):
            ind = []
            for p in range(10):
                ind.append(random.uniform(self.xl[p], self.xu[p]))
            pop.append(ind)

        return pop
    
    def reference_point(self):
        return

    def initialization(self):
        return
                

ag = AG_MOM(1, 1, 1, 1, 1, 1, 1)
ag.initialization()
