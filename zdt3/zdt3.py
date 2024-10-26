import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ZDT3():
    def __init__(self, N, G, mut_op, cross_op, T, xl=0., xu=1.):
        self.N = N
        self.G = G
        self.mut_op = mut_op
        self.cross_op = cross_op
        self.T = T
        self.xl = xl
        self.xu = xu
        self.neighbors_size = math.floor(T * N)
        self.p = 30
        self.delta = random.randint(0,self.p - 1)
        self.m = 2
        self.F = 0.5
        self.pr = 1/self.p
        self.SIG = 20
        self.vectors = []
        self.neighbors = []
        self.pop = []
        self.fobj = []
        self.z = []

    def create_vectors(self):
        vectors = []
        dst = 0
        while dst < 1:
            vector = [0+dst, 1-dst]
            vectors.append(vector)
            dst += 1/self.N
        self.vectors = np.array(vectors)

    def find_neightbors(self):
        N = self.N
        self.create_vectors()
        dsts = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dsts[i, j] = np.linalg.norm(np.array(self.vectors[i]) - np.array(self.vectors[j]))

        neighbors = []

        for i in range(N):
            nearest_indexes = np.argsort(dsts[i])[:self.neighbors_size]
            neighbors.append(nearest_indexes)

        self.neighbors = neighbors

    def initialize_pop(self):
        pop = []
        for i in range(self.N):
            ind = []
            for p in range(self.p):
                ind.append(random.uniform(self.xl, self.xu))
            pop.append(ind)

        self.pop = pop
    
    def evaluate_indv(self, indv):
        tmp = 0
        obj = [0, 0]
        obj[0] = indv[0]
        for i in range(1, self.p):
            tmp += indv[i]

        g = 1 + ((9 * tmp) / (self.p - 1))
        h = 1 - math.sqrt(indv[0] / g) - (indv[0] / g) * math.sin(10 * math.pi * indv[0])
        obj[1] = g * h
        return obj

    def evaluate_pop(self):
        fobj = []
        for indv in self.pop:
            obj = self.evaluate_indv(indv)
            fobj.append(obj)

        self.fobj = fobj

    def reference_point(self):
        z = []
        for i in range(self.m):
            zi = min(self.fobj, key=lambda x:x[i])[i]
            z.append(zi)
        self.z = z
        
    def overwrite_file_dominant_solutions(self, solutions):
        with open("./zdt3/dominant_solutions.dat", "w") as archivo:
            for sol in solutions:
                archivo.write(f"{sol[0]:.6f}\t{sol[1]:.6f}\n")
    
    def save_dominant_solutions(self):
        res = []
        for sol1 in self.fobj:
            aux = True
            for sol2 in self.fobj:
                if sol1[0] > sol2[0] and sol1[1] > sol2[1]:
                    aux = False
                    break
            if aux:
                res.append(sol1)
    
        self.overwrite_file_dominant_solutions(res)
                
    def initialization(self):
        self.create_vectors()
        self.find_neightbors()
        self.initialize_pop()
        self.evaluate_pop()
        self.reference_point()
        self.save_dominant_solutions()
        with open("./zdt3/zdt3_all_popm.out", "w") as archivo:
            for sol in self.fobj:
                archivo.write(f"{sol[0]:.6f}\t{sol[1]:.6f}\t{0.:.6f}\n")
        
    def g_te(self, obj_indv, vector):
        return max([vector[i]*abs(obj_indv[i] - self.z[i]) for i in range(self.m)])
    
    def read_dat(self, file_name):
        res = []
        with open(file_name, 'r') as file:
            for i in file:
                coordinates = i.split()
                res.append([float(coordinates[0]), float(coordinates[1])])
        return res
    
    def update_dominant_solutions(self, obj_indv):
        dominant_solutions = self.read_dat("./zdt3/dominant_solutions.dat")
        aux = True
        for sol in dominant_solutions:
            if sol[0] >= obj_indv[0] and sol[1] >= obj_indv[1]:
                dominant_solutions.remove(sol)
            elif sol[0] <= obj_indv[0] and sol[1] <= obj_indv[1]:
                aux = False
        if aux:
            dominant_solutions.append(obj_indv)
        self.overwrite_file_dominant_solutions(dominant_solutions)

    def reproduction(self):
        for i in range(self.N):
            indv = np.array(self.pop[i])
            # Metodo de curce
            neighbors_indexes = random.sample(self.neighbors[i].tolist(), k=3)
            neighbors = [np.array(self.pop[m]) for m in neighbors_indexes]
            v = neighbors[0] + self.F * (neighbors[1] - neighbors[2])
            for k in range(self.p):
                if random.random() < self.cross_op or i == self.delta:
                    indv[k] = v[k]
            # Mutacion gaussiana
            if random.random() < self.pr:
                sigma = (self.xu - self.xl)/self.SIG
                indv = indv + np.random.normal(0, sigma, size=30)
            # Asegurar de que no se sobrpasa el espacio de busqueda definido mediante rebote
            indv = np.array([-x if x < 0 else (2 - x if x > 1 else x) for x in indv]) 
            obj_indv = self.evaluate_indv(indv)
            for m in range(self.m):
                if obj_indv[m] < self.z[m]:
                    self.z[m] = obj_indv[m] 
            # Actualizar fichero de las soluciones no dominadas
            self.update_dominant_solutions(obj_indv)
            for j in self.neighbors[i]:
                obj_neighbor = self.fobj[j]
                vector = self.vectors[j]
                if self.g_te(obj_indv, vector) <= self.g_te(obj_neighbor, vector):
                    self.pop[j] = indv.tolist()
                    self.fobj[j] = obj_indv

    
    def read_dat_separate_coordinates(self, file_name):
        x = []
        y = []
        with open(file_name, 'r') as file:
            for i in file:
                coordinates = i.split()
                x.append(float(coordinates[0]))
                y.append(float(coordinates[1]))
        return x, y
    
    def read_dat_separate_coordinates_nsga(self, file_name):
        x = []
        y = []
        with open(file_name, 'r') as file:
            next(file)
            next(file)
            for i in file:
                coordinates = i.split()
                x.append(float(coordinates[0]))
                y.append(float(coordinates[1]))
        return x, y
    
    def separate_coordinates(self):
        x = []
        y = []
        for i in self.fobj:
            x.append(i[0])
            y.append(i[1])
        return x, y
    

    def ag_mobj(self):
        self.initialization()
        
        fig, ax = plt.subplots()
        
        x_pf, y_pf = self.read_dat_separate_coordinates('./zdt3/PF.dat')
        pareto_plot = ax.scatter(x_pf, y_pf, color='green', label='Pareto front', marker='o')

        x, y = self.separate_coordinates()
        pop_plot = ax.scatter(x, y, color='blue', label='F(x)', marker='o')

        ax.set_xlabel("f1(x)")
        ax.set_ylabel("f2(x)")
        ax.legend()

        iteration_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        def update(frame):
            self.reproduction()
            if frame != 0:
                # Añadir soluciones al archivo all_popm
                with open("./zdt3/zdt3_all_popm.out", "a") as archivo:
                    for sol in self.fobj:
                        archivo.write(f"{sol[0]:.6f}\t{sol[1]:.6f}\t{0.:.6f}\n")
                
            if frame == 99:
                with open("./zdt3/zdt3_final_popp.out", "w") as archivo:
                    for i in range(self.N):
                        sol = self.fobj[i]
                        archivo.write(f"{sol[0]:.6f}\t{sol[1]:.6f}\t{0.:.6f}\n")
                        
            x, y = self.separate_coordinates()
            pop_plot.set_offsets(np.c_[x, y])
            iteration_text.set_text(f"Iteración: {frame + 1}")
            
            return pop_plot, iteration_text

        anim = FuncAnimation(fig, update, frames=self.G, interval=100, repeat=False)

        plt.show()
        return self.pop


ag = ZDT3(100, 100, 0.03, 0.5, 0.3, 0., 1.)
ag.ag_mobj()
