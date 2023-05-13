# -*- coding: utf-8 -*-
"""
Created on Sun Apr 9 23:35:57 2023

@author: aroy
"""

import numpy as np
import matplotlib.pyplot as plt
import time


tic = time.perf_counter()

#Spatial discretization
#2nd order forward discrete laplacian function      #using np.roll
def laplacian(M):
    L = -4*M
    L += np.roll(M,(0,-1), (0,1))   #right neighbour
    L += np.roll(M,(0,+1), (0,1))   #left neighbour
    L += np.roll(M,(-1,0), (0,1))   #top neighbour
    L += np.roll(M,(+1,0), (0,1))   #bottom neighbour
    return L

#time derivative of u field (dynamical equation)

def time_derivative(u):
    Lu = laplacian(u)

    s = kf*((1-u)/2)-kb*((1+u)/2)              #Reaction
    mu = Lu + (1/(d**2))*(u-u**3) + s

    #Apply time derivative formula
    diff_u = -(k/tau)*laplacian(mu)
    return diff_u

# time update (Forward Euler time stepping)

#def time_update(u):
#    diff_u = time_derivative(u)
#    u += diff_u* dt
#    return u

# time update (Runge-Kutta 4th order)
def time_update(u_0):
    k1 = time_derivative(u_0)
    u_1 = u_0+k1*dt/2
    k2 = time_derivative(u_1)
    u_2 = u_0+k2*dt/2
    k3 = time_derivative(u_2)
    u_3 = u_0+k3*dt
    k4 = time_derivative(u_3)
    arr = u_0 + 1/6 * dt * (k1+2*k2+2*k3+k4)
    return arr

N = 100
R = 10


x = np.linspace(0,100,N+1)
y = np.linspace(0,100,N+1)
xx, yy = np.meshgrid(x, y, sparse = True)
rr = np.sqrt((xx-40*np.ones((1,N+1)))**2+(yy-40*np.ones((N+1,1)))**2)
#print (xx.shape,yy.shape,zz.shape)

# single droplet initialization
def initialize_droplet(box_length):
    uarr = np.tanh((R-rr)/d) + 0.2*np.random.random((N+1,N+1))         #single droplet
    return uarr
    
#Parameters
k = 0.1
d = 1.0
tau=1
ntime = 500000
box_length = 100
dt = 0.1
#ini_dens = -0.1                                 # for Forward Euler trial(without reaction code)
#var = .1
kf = 0.0005
kb = 0.005
# dx = 1
#initialization of our model
u = initialize_droplet(box_length)

# Evolving the system through time
for i in range(ntime):
    u_n=time_update(u)
    u=u_n
toc = time.perf_counter()
t = toc-tic
print(t)
print(np.amax(u))
print(np.amin(u))

#uncomment savefig command to save figure
def main(name):
    fig,ax = plt.subplots()
    v = ax.imshow(u, interpolation='none', cmap='viridis')
    fig.colorbar(v, ax=ax)
    #plt.title('%s, %s, %s, %s' %(k/tau,d,ini_dens,ntime))          # for random initialization
    plt.title('%s, %s, %s' %(k/tau,d,ntime))
    fig.savefig(name, dpi = 1200)

main('kbytau=%s_d=%s_time=%s_dt=0.1_final_box=100_single_droplet_RK4_rxn_kf=5e-4_kb=5e-3_R=10_12th_May.png' %(k/tau,d,ntime))
