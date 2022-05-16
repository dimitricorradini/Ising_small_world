import numpy as np
import matplotlib.pyplot as plt

#define extrema of interval of interest
t_0 = 0
t_1 = 4
#number of subdivisions of interval [t_0,t_1]
n = 500






#definisco il mio sistema di eq
def f(S, c):
	return (S-1+np.exp(-c*S))


#definisco la df/dS
def df(S, c):
	return (1-c*np.exp(-c*S))
	

#matrice dei risultati
S_zero = np.zeros((n+1,))

#algoritmo di Newton
for q in range(0, n+1):
	S_guess = 1

	for i in range(0, 1000):
		S_guess = S_guess - f(S_guess, q*(t_1-t_0)/n+t_0)/df(S_guess, q*(t_1-t_0)/n+t_0)
	S_zero[q]=S_guess
	print(q)

#stampo
colors = ['#004646']

plt.plot(np.arange(t_0,t_1+(t_1-t_0)/n,step=(t_1-t_0)/n),S_zero, color=colors[0])


plt.show()
