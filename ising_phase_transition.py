import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation


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
        print(temperature)
        for i in range(int(self.grid.shape[0])):
            for j in range(int(self.grid.shape[1])):
                a = np.random.randint(0, self.grid.shape[0])
                b = np.random.randint(0, self.grid.shape[1])
                #mag_sum = np.sum(magnetic*self.grid)
                cost =  2 * self.grid[a,b] * (self.grid[(a+1)%self.grid.shape[0],b] + self.grid[a,(b+1)%self.grid.shape[1]] + self.grid[(a-1)%self.grid.shape[0],b] + self.grid[a,(b-1)%self.grid.shape[1]]+ mu * magnetic[a,b])
                if cost < 0 :
                    self.grid[a,b] *= -1
                elif np.random.uniform() < np.exp(-cost/(k*temperature)):
                    self.grid[a,b] *= -1

    def mag(self):
        count = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 1:
                    count += 1
        return((2*count-self.height * self.width)/(self.height * self.width))

             

model1 = Ising(100, 100, 1)

mag_means = np.array([])
for T in np.arange(0,3,0.15):
    mag = np.array([])
    for j in range(0,1000):
        model1.Evolve(0.0001+T, np.zeros((model1.height,model1.width)))
        print(j, ' 1')
    for j in range(0,1000):
        model1.Evolve(0.0001+T, np.zeros((model1.height,model1.width)))
        print(j, ' 2')
        if j % 20 == 0:
           mag = np.append(mag, model1.mag())
    mag_means= np.append(mag_means, np.mean(mag))




plt.plot(np.arange(0, 3, 0.15), mag_means, color = '#004646')

plt.show()
