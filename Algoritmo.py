import numpy as np
import random
import math


class AG_MOBJ():
    def __init__(self, N, G, mut_op, cross_op, T, xl=0, xu=1):
        self.N = N
        self.G = G
        self.mut_op = mut_op
        self.cross_op = cross_op
        self.T = T
        self.xl = xl
        self.xu = xu
        self.p = 30
        self.m = 2
        self.vectors = []
        self.neightbors = []
        self.pop = []
        self.fobj = []
        self.z = []

    def create_vectors(self):
        vectors = []
        dst = 1/self.N
        while dst < 1:
            vector = [0+dst, 1-dst]
            vectors.append(vector)
            dst += dst
        self.vectors = vectors

    def find_neightbors(self):
        N = self.N
        vectors = self.create_vectors()
        dsts = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                dsts[i, j] = np.linalg.norm(vectors[i] - vectors[j])

        neighbors = []

        for i in range(N):
            nearest_indexes = np.argsort(dsts[i])[:self.T]
            neighbors.append(nearest_indexes)

        self.neightbors = neighbors

    def initialize_pop(self):
        pop = []
        for i in range(self.N):
            ind = []
            for p in range(self.p):
                ind.append(random.uniform(self.xl, self.xu))
            pop.append(ind)

        self.pop = pop

    def evaluate_pop(self):
        fobj = []
        for indv in self.pop:
            tmp = 0
            obj = [0, 0]
            obj[0] = indv[0]
            for i in range(1, self.N):
                tmp += self.pop[i]

            g = 1 + ((9 * tmp) / (self.N - 1))
            h = 1 - math.sqrt(indv[0] / g) - (indv[0] / g) * math.sin(10 * math.pi * indv[0])
            obj[1] = g * h
            fobj.append(obj)

        self.fobj = fobj

    def reference_point(self):
        z = []
        for i in range(self.m):
            zi = min(self.fobj, key=lambda x:x[i])[i]
            z.append(zi)
        self.z = z

    def initialization(self):
        self.create_vectors()
        self.find_neightbors()
        self.initialize_pop()
        self.evaluate_pop()
        self.reference_point()


ag = AG_MOBJ(1, 1, 1, 1, 1, 1, 1)
ag.initialization()
