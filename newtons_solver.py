import numpy as np
import matplotlib.pyplot as plt

#define extrema of interval of interest
t_0 = -1
t_1 = 1
#number of subdivisions of interval [t_0,t_1]
n = 300
T = 0.4

#necessary coefficients and parameters
c= np.array([[0.1, 0.2, 0.1, 0.3],[0.2,0.05,0.3,0.2],[0.3,0.2,0.1,0.2],[0.1,0.15,0.1,0.2]])
c_1 = np.array([0.9, 0.3, 0.3, 0.5])
#c_1: vettore colonna dei coeff. di m_1
J= np.array([[0.1, 0.2, 0.1, 0.3],[0.2,0.05,0.3,0.2],[0.3,0.2,0.1,0.2],[0.1,0.15,0.1,0.2]])
J_1 = np.array([0.9, 0.3, 0.3, 0.5])


#couplings

l = c_1.size +1
#number of population subgroups

#connettività efficaci
if T == 0:
	c_t =c
	c_1_t =c_1
else:
	c_t = c*np.tanh(1/T*J)
	c_1_t = c_1*np.tanh(1/T*J_1)  

#stampo det(1-c), utile per la c.d. 'percolation' (isteresi)

print("det = ", np.linalg.det(np.identity(l-1)-c_t))


#definisco il mio sistema di eq
def f(m, m_1):
	return (np.tanh(c_1_t*m_1+c.dot(m))-m)


#definisco lo jacobiano di f
def J(m, m_1):
	P = c*(1/((np.cosh(c_1_t*m_1+c_t.dot(m))*(np.cosh(c_1_t*m_1+c_t.dot(m))))))-np.identity(l-1)
	return (P)

#matrice dei risultati
m_zero = np.zeros((2*l-1,n+1))

#algoritmo di Newton
for q in range(0, n+1):
	m_zeroes = np.ones(l-1,)

	for i in range(0, 50):
		m_zeroes = m_zeroes - np.linalg.inv(J(m_zeroes, q*(t_1-t_0)/n+t_0)).dot(f(m_zeroes, q*(t_1-t_0)/n+t_0))
		for j in range(m_zeroes.size):
			#if m_zeroes[j] < 0:
				#m_zeroes[j] = 0
			if m_zeroes[j] > 1:
				m_zeroes[j] = 0
	m_zero[0:l-1,q]=m_zeroes
	print(q)
for q in range(0, n+1):
	m_zeroes = -1*np.ones(l-1,)

	for i in range(0, 50):
		m_zeroes = m_zeroes - np.linalg.inv(J(m_zeroes, q*(t_1-t_0)/n+t_0)).dot(f(m_zeroes, q*(t_1-t_0)/n+t_0))
		for j in range(m_zeroes.size):
			if m_zeroes[j] > 0:
				m_zeroes[j] = 0
			if m_zeroes[j] < -1:
				m_zeroes[j] = 0
	m_zero[l:2*l-1,q]=m_zeroes
	print(q)
t_back = []
t_forward = []
n_back = []
n_forward = []

for p in range(l, 2*l-1):
	for q in range(n+1):
		if m_zero[p,q] == 0 or m_zero[p,q+1]-m_zero[p,q]>0.1 or q == n or m_zero[p,q+1]-m_zero[p,q]<0:
			n_forward.append(q)
			t_forward.append(t_0+q*(t_1-t_0)/n)
			break

#stampo
colors = ['#004646', '#309898', '#ff9f00', '#E0350B']
line = []
print(n_forward[0])
for j in range(0,l-1):
    line.append(plt.plot(np.arange(-t_forward[j], t_1, step=(t_1-t_0)/n),m_zero[j,(n-n_forward[j]):n], label = "$m^{(%s)}$" % (j+2), color=colors[j]))

for j in range(0,l-1):
    plt.plot(np.arange(t_0,t_forward[j]+(t_1-t_0)/n, step=(t_1-t_0)/n), m_zero[j+l,0:n_forward[j]+1], color=colors[j])

plt.legend()
plt.show()
