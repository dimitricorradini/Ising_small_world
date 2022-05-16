import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap


def clear():
    _ = call('clear' if os.name =='posix' else 'cls')

state = [1, -1]

k = 1
mu = 0.01

class Ising():

    def __init__(self, height, width, init_prob):
        count = 0

        self.height = height
        self.width = width
        self.grid = np.random.choice(a = state, size = (height, width), p = [init_prob,1-init_prob])
        self.J_SG = np.random.normal(0, 2, size = (self.grid.shape[0],self.grid.shape[1]))
        self.history = []

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                print(i, " ", j)
                if self.grid[i, j] == 1:
                    count += 1

        self.history.append((2*count-height*width)/(height*width))
    

    def set_ones(self):
        count = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.grid[i, j] = 1 

        self.history[0] = self.height * self.width
    

    def set_minus_ones(self):
        count = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.grid[i, j] = -1 

        self.history[0] = -self.height * self.width
    
    def Evolve(self, temperature = 0, magnetic = np.zeros((100,100))):
        count = 0
        for i in range(int(self.grid.shape[0])):
            for j in range(int(self.grid.shape[1])):
                print(i, " ", j)
                a = np.random.randint(0, self.grid.shape[0])
                b = np.random.randint(0, self.grid.shape[1])
                #mag_sum = np.sum(magnetic*self.grid)
                cost =  2 * self.grid[a,b] * (self.grid[(a+1)%self.grid.shape[0],b] + self.grid[a,(b+1)%self.grid.shape[1]] + self.grid[(a-1)%self.grid.shape[0],b] + self.grid[a,(b-1)%self.grid.shape[1]]+ mu * magnetic[a,b])
                if cost < 0 :
                    self.grid[a,b] *= -1
                elif np.random.uniform() < np.exp(-cost/(k*temperature)):
                    self.grid[a,b] *= -1

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 1:
                    count += 1
        self.history.append((2*count-self.height * self.width)/(self.height * self.width))

                

model1 = Ising(100, 100, 0.5)
model2 = Ising(20, 20, 0)


def generate_data():
    model1.Evolve(0.26, np.ones((model1.height,model1.width)))
    return model1.grid


def update(data):
    mat.set_data(data)
    return mat 

def data_gen():
    while True:
        yield generate_data()

fig, ax = plt.subplots()

cmap = ListedColormap(['#ff9f00', '#E0350B'])
mat = ax.matshow(generate_data(), cmap = cmap)

ani = animation.FuncAnimation(fig, update, data_gen, interval=100,
                             save_count=50)


ax = plt.gca()

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

plt.plot(model1.history, color = '#004646')

plt.show()
