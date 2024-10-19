import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CF6():
    def __init__(self, N, G, mut_op, cross_op, T, n, xl1=0., xu1=1., xli = -2., xui = 2.):
        self.N = N
        self.G = G
        self.n = n
        self.mut_op = mut_op
        self.cross_op = cross_op
        self.T = T
        self.xl1 = xl1
        self.xu1 = xu1
        self.xli = xli
        self.xui = xui
        self.neighbors_size = math.floor(T * N)
        self.delta = random.randint(0,self.n - 1)
        self.m = 2
        self.F = 0.5
        self.pr = 1/self.n
        self.SIG = 20
        self.J1 = [j for j in range(3, self.n + 1, 2)]
        self.J2 = [j for j in range(2, self.n + 1, 2)]
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
            for p in range(self.n):
                if p == 0:
                    ind.append(random.uniform(self.xl1, self.xu1))
                else:
                    ind.append(random.uniform(self.xli, self.xui))
            pop.append(ind)

        self.pop = pop
    
    def evaluate_indv(self, indv):
        obj = [0, 0, 0]
        
        yj1 = 0
        for j in self.J1:
            yj1 += (indv[j-1] - 0.8 * indv[0] * math.cos(6 * math.pi * indv[0] + (j * math.pi)/self.n)) ** 2
        obj[0] = indv[0] + yj1
        
        yj2 = 0
        for j in self.J2:
            yj2 += (indv[j-1] - 0.8 * indv[0] * math.cos(6 * math.pi * indv[0] + (j * math.pi)/self.n)) ** 2
        obj[1] = (1 - indv[0]) ** 2 + yj2 
        
        const = 0
        const1 = indv[1] - 0.8 * indv[0] * math.sin(6 * math.pi * indv[0] + (2 * math.pi)/self.n) - np.sign(0.5 * (1 - indv[0]) - (1 - indv[0]) ** 2) * math.sqrt(abs(0.5 * (1 - indv[0]) - (1 - indv[0]) ** 2))
        # print(const1)
        if const1 < 0:
            const += const1
        const2 = indv[3] - 0.8 * indv[0] * math.sin(6. * math.pi * indv[0] + 4. * math.pi/self.n) - np.sign(0.25 * math.sqrt(1. - indv[0]) - 0.5 * (1. - indv[0])) * math.sqrt(abs(0.25 * math.sqrt(1. - indv[0]) - 0.5 * (1. - indv[0])))
        # print(const2)
        # print(np.sign(0.25 * math.sqrt(1 - indv[0]) - 0.5 * (1 - indv[0])))
        # print(np.sign(0.25 * math.sqrt(1. - indv[0]) - 0.5 * (1. - indv[0])) * math.sqrt(abs(0.25 * math.sqrt(1. - indv[0]) - 0.5 * (1. - indv[0]))))
        # print(indv[3] - 0.8 * indv[0] * math.sin(6. * math.pi * indv[0] + 4. * math.pi/self.n))
        if const2 < 0:
            const += const2
        obj[2] = const
        
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
        with open("./cf6/dominant_solutions.dat", "w") as archivo:
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
        with open("./cf6/cf6_all_popm.out", "w") as archivo:
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
        dominant_solutions = self.read_dat("./cf6/dominant_solutions.dat")
        aux = True
        for sol in dominant_solutions:
            if sol[0] >= obj_indv[0] and sol[1] >= obj_indv[1]:
                dominant_solutions.remove(sol)
            elif sol[0] <= obj_indv[0] and sol[1] <= obj_indv[1]:
                aux = False
        if aux:
            dominant_solutions.append(obj_indv)
        self.overwrite_file_dominant_solutions(dominant_solutions)
        
    def check_bounds(self, indv):
        for i in range(self.n):
            if i == 0:
                if indv[i] < 0:
                    indv[i] = -indv[i]
                elif indv[i] > 1:
                    indv[i] = 2 - indv[i]
            else:
                if indv[i] < -2:
                    indv[i] = -2
                elif indv[i] > 2:
                    indv[i] = 2
        return indv

    def reproduction(self):
        for i in range(self.N):
            indv = np.array(self.pop[i])
            # Metodo de curce
            neighbors_indexes = random.sample(self.neighbors[i].tolist(), k=3)
            neighbors = [np.array(self.pop[m]) for m in neighbors_indexes]
            v = neighbors[0] + self.F * (neighbors[1] - neighbors[2])
            for k in range(self.n):
                if random.random() < self.cross_op or i == self.delta:
                    indv[k] = v[k]
            # Mutacion gaussiana
            if random.random() < self.pr:
                sigma = (self.xui - self.xli)/self.SIG
                indv = indv + np.random.normal(0, sigma, size=self.n)
            # Asegurar de que no se sobrpasa el espacio de busqueda definido mediante rebote
            indv = self.check_bounds(indv)
            obj_indv = self.evaluate_indv(indv)
            for m in range(self.m):
                if obj_indv[m] < self.z[m]:
                    self.z[m] = obj_indv[m] 
            # Actualizar fichero de las soluciones no dominadas
            self.update_dominant_solutions(obj_indv)
            const_indv = obj_indv[2]
            for j in self.neighbors[i]:
                obj_neighbor = self.fobj[j]
                const_neighbor = obj_neighbor[2]
                vector = self.vectors[j]
                if const_neighbor == 0 and const_indv == 0:
                    if self.g_te(obj_indv, vector) <= self.g_te(obj_neighbor, vector):
                        self.pop[j] = indv.tolist()
                        self.fobj[j] = obj_indv
                elif const_neighbor < 0 and const_indv == 0:
                    self.pop[j] = indv.tolist()
                    self.fobj[j] = obj_indv
                elif const_neighbor < 0 and const_indv < 0:
                    if const_indv > const_neighbor:
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
        
        x_pf, y_pf = self.read_dat_separate_coordinates('./cf6/PF.dat')
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
                with open("./cf6/cf6_all_popm.out", "a") as archivo:
                    for sol in self.fobj:
                        archivo.write(f"{sol[0]:.6f}\t{sol[1]:.6f}\t{0.:.6f}\n")
            x, y = self.separate_coordinates()
            pop_plot.set_offsets(np.c_[x, y])
            iteration_text.set_text(f"Iteración: {frame + 1}")
            
            return pop_plot, iteration_text

        anim = FuncAnimation(fig, update, frames=self.G, interval=100, repeat=False)

        plt.show()
        return self.pop


ag = CF6(100, 100, 0.03, 0.5, 0.3, 4, 0., 1., -2., 2.)
indv = np.array([0.56459503,0.01702003,-0.1605145,0.27745099])
ag.ag_mobj()
