# -*- coding: utf-8 -*-
"""
Created on Sun Apr 9 23:35:57 2023

@author: aroy
"""
#With this code we are simulating the differential equation shown in Section 1 of the Phys 230 Project pdf.
#Specifically, this is liquid-liquid phase separation for two components on a flat surface with a density array ranging from
#from -1 to 1. We also want to compare the simulation time between 4th order Runge Kutta and Forward Euler.

#Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
from numba import njit #Used to speed up loops
import time
import matplotlib.animation as animation
from matplotlib.colors import Normalize

tic = time.perf_counter() #To find initial time of simulation

# Parameters
k = 0.1
d = 1.0
tau=1
ntime = 5000
box_length = 100
dt = 0.1
ini_dens = -0.1                                 # for Forward Euler trial(without reaction code)
var = .1
#kf = 0.1
#kb = 0.9
N = 100
R = 10

tcls = '' #Specifies if you want to generate an array corresponding to simulation time at various steps
time_vals = [] # Array to hold simulation time at various time steps
tcalc_iter=1000 #Number of steps between each simulation time check

@njit(fastmath=True)       #use njit(just in time)decorator
    
def laplacian(Z):
    L=np.zeros((box_length,box_length)) #Initializing array for laplacian
    for i in range(box_length):
       iup1 = (i+1)%box_length #These relations are used to enforce PBC, since going over lattice length goes back to 1
       idown1 = (i-1)%box_length
       iup2 = (i+2)%box_length
       idown2 = (i-2)%box_length
       for j in range(box_length):
           jup1 = (j+1)%box_length
           jdown1 = (j-1)%box_length
           jup2 = (j+2)%box_length
           jdown2 = (j-2)%box_length
           L[i,j]+= (-Z[iup2,j]+16*Z[iup1,j]-30*Z[i,j]+16*Z[idown1,j]-Z[idown2,j] -Z[i,jup2]+16*Z[i,jup1]-30*Z[i,j]+16*Z[i,jdown1]-Z[i,jdown2])/12
           
    return L

#Finding time derivative for density arrays
@njit(fastmath=True)
def time_derivative(u):
    Lu = laplacian(u)
    #s = kf*((1-u)/2)-kb*((1+u)/2)              #Reaction
    mu = Lu + (1/(d**2))*(u-u**3)
    #Apply time derivative formula
    diff_u = -(k/tau)*laplacian(mu)
    return diff_u

# Time update method-I ( Forward Euler timestepping method)
#@njit(fastmath=True)
#def time_update(u):
#    diff_u = time_derivative(u)
#    u += diff_u* dt
#    return u

# Time update method-II (Runge Kutta 4th order method)
@njit(fastmath=True)
def time_updaterk(u_0):
    k1 = time_derivative(u_0)
    u_1 = u_0+k1*dt/2
    k2 = time_derivative(u_1)
    u_2 = u_0+k2*dt/2
    k3 = time_derivative(u_2)
    u_3 = u_0+k3*dt
    k4 = time_derivative(u_3)
    arr = u_0 + 1/6 * dt * (k1+2*k2+2*k3+k4)
    return arr

#Initializing uarray Function such that it is between -1 and 1 (with slight bias towards -1)
@njit(fastmath=True)
def initialize(length,mu,sigma):
    while True:
        uarr = np.random.normal(mu, sigma, (length,length))
        if np.amax(uarr)<=1 and np.amin(uarr)>=-1:
            break
    return uarr

# Initializing single droplet (tanh function)

#x = np.arange(0,100,101,dtype = int)
#y = np.arange(0,100,101,dtype=int)
#r = np.sqrt(x**2+y**2)
#xx, yy = np.meshgrid(x, y)
#rr = np.sqrt((xx-40*np.ones((N+1,N+1)))**2+(yy-40*np.ones((N+1,N+1)))**2)
#print (xx.shape,yy.shape,zz.shape)
#@njit(fastmath=True)
#def initialize_droplet(N):
#    uarr = np.tanh((R-rr)/d) + 0.2*np.random.random((N+1,N+1))         #single droplet
#    return uarr
    
#Creating start function that evolves the system through time. Can uncomment/comment tstep(u_arr)
#to switch between the Runge Kutta and Forward Euler methods

def start(u_arr,tcalc):
    #ngraph=0 #Numbering for printed out graphs if needed
    for i in range(ntime):
        #u_arrn=time_update(u_arr)
        u_arrn=time_updaterk(u_arr)
        if tcalc=='yes' and i%tcalc_iter==0: #If wanted, this tells the code to determine simulation time throughout simulation
            toc = time.perf_counter()
            time_vals.append(toc-tic)
        u_arr=u_arrn
    return u_arr

#Creating main function to run simulation

def main():
    u_arr=initialize(box_length,ini_dens,var)             # for random initialization
#    u_arr=initialize_droplet(box_length)                   # for single droplet initialization
    u_arrf=start(u_arr,tcls)
    return u_arrf

final = main()
#print(np.amax(final))
#print(np.amin(final))

# print key things about simulation time_vals
toc = time.perf_counter()
#print('Simulation time array:',time_vals) #Print out array of simulation time values if needed
print('The total time of simulation was:',toc-tic)

# create plotting Function
def plotting(name):
    fig,ax = plt.subplots()
    v = ax.imshow(final, interpolation='gaussian', cmap='viridis')
    fig.colorbar(v, ax=ax)
    #plt.title('%s, %s, %s, %s' %(k/tau,d,ini_dens,ntime))
    plt.title('%s, %s, %s' %(k/tau,d,ntime))
    fig.savefig(name, dpi = 1200)

#plotting('kbytau=%s_d=%s_inidens=%s_time=%s_dt=0.1_init_single_droplet_rxn_RK4_numba_1st_May.png' %(k/tau,d,ini_dens,ntime))
plotting('kbytau=%s_d=%s_time=%s_dt=0.1_init_random_init_Euler_numba_k=0.1_12th_May.png' %(k/tau,d,ntime))


