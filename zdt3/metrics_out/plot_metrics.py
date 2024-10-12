import matplotlib.pyplot as plt

def read_dat_separate_coordinates(file_name):
        x = []
        y = []
        with open(file_name, 'r') as file:
            for i in file:
                coordinates = i.split()
                x.append(float(coordinates[0]))
                y.append(float(coordinates[1]))
        return x, y

def plot_spacing():
    
    fig, ax = plt.subplots()
    
    x1, y1 = read_dat_separate_coordinates('spacing.out')
    pareto_plot = ax.plot(x1, y1, color='green', label='C1')
    
    x2, y2 = read_dat_separate_coordinates('spacing2.out')
    pareto_plot = ax.plot(x2, y2, color='red', label='C2')
    
    ax.set_xlabel("Generations")
    ax.set_ylabel("Spacing")
    ax.legend()
    
    plt.show()
    
def plot_hypervol():
    
    fig, ax = plt.subplots()
    
    x1, y1 = read_dat_separate_coordinates('hypervol.out')
    pareto_plot = ax.plot(x1, y1, color='green', label='C1')
    
    x2, y2 = read_dat_separate_coordinates('hypervol2.out')
    pareto_plot = ax.plot(x2, y2, color='red', label='C2')
    
    ax.set_xlabel("Generations")
    ax.set_ylabel("Hypervol")
    ax.legend()
    
    plt.show()
    
def plot_extent():
    
    fig, ax = plt.subplots()
    
    x1, y1 = read_dat_separate_coordinates('extent.out')
    pareto_plot = ax.plot(x1, y1, color='green', label='C1')
    
    x2, y2 = read_dat_separate_coordinates('extent2.out')
    pareto_plot = ax.plot(x2, y2, color='red', label='C2')
    
    ax.set_xlabel("Generations")
    ax.set_ylabel("extent")
    ax.legend()
    
    plt.show()
    
plot_spacing()
plot_hypervol()
plot_extent()