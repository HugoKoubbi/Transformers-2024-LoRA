import numpy as np
import scipy as sp
import imageio
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from datetime import datetime
from multiprocessing import Pool
np.random.seed(42)
def get_dynamics(z_curr, attention, V, i):
    """
    - Returns: the dynamics z'(t) = (z_1'(t), ... , z_n'(t)) at some time-step t.
    """
    
    dlst = np.array([attention[i][j]*np.matmul(V, z_curr[j]-z_curr[i]) for j in range(n)])
    return np.sum(dlst, axis=0)

def transformer(T, dt, n, d, A, V, x0):
    """
    - Returns: the evolution of z = (z_1, ..., z_n) over time.
    """
    
    num_steps = int(T/dt)+1
    z = np.zeros(shape=(n, num_steps, d))
    z[:, 0, :] = x0
    integration_time = np.linspace(0, T, num_steps)

    for l, t in enumerate(integration_time):
        if l < num_steps - 1:
            # Attention matrix
            attention = [[1/np.sum([np.exp(np.dot(np.matmul(np.matmul(A, sp.linalg.expm(V*t)), z[i][l]), np.matmul(np.matmul(A, sp.linalg.expm(V*t)), z[k][l]-z[j][l]))) for k in range(n)]) for j in range(n)] for i in range(n)]
            
            z_next = np.zeros((n, d))
            for i in range(n):
                k1 = dt * get_dynamics(z[:, l, :], attention, V, i)
                k2 = dt * get_dynamics(z[:, l, :] + k1 / 2, attention, V, i)
                k3 = dt * get_dynamics(z[:, l, :] + k2 / 2, attention, V, i)
                k4 = dt * get_dynamics(z[:, l, :] + k3, attention, V, i)
                
                z_next[i] = z[i][l] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
            z[:, l+1, :] = z_next
    return z

def calculate_distance_from_viewing_direction(point):
    viewing_direction = np.array([1, 0, 0])
    return np.linalg.norm(point - viewing_direction)

####### Distance to the clusters
def distance_to_clusters(clusters,z,t): 
    d=0
    for i in range(n):
        d_i=100
        for x in clusters:
            d_i=min(d_i,np.linalg.norm(z[i, t, :]-x))
        d=max(d,d_i)
    return(d)
####### Time bifurcation
def time_bifurcation(clusters,z,T,dt,delta):
    num_steps = int(T/dt)+1
    integration_time = np.linspace(0, T, num_steps) 
    x=0
    x2=0
    for t in integration_time:
        if distance_to_clusters(clusters,z,int(t/dt))<delta and x==0:
            x=t
        if distance_to_clusters(clusters,z,int(t/dt))>delta and x!=0:
            x2=t
            break
    return(x,x2)

####### Hyperparameters of the experience
geometries = ["1rankattention","polytope", "hyperplanes", "codimension-k", "hyperplanes x polytope","LoRA","V_finetune","V_finetune_3","degenerate"]
L_clus=[]
L_phase=[]
Time = 50
dt = 0.1
d = 2
n=20
num_steps = int(Time/dt)+1
x0 = np.random.uniform(low=-1, high=1, size=(n, d))
integration_time = np.linspace(0, Time, num_steps)
movie = False
conv = False
show_polytope = False
V=np.eye(d)
A = np.eye(d)
z=transformer(Time, dt, n, d, A, V, x0)
delta=0.1
clusters=[ z[i, -1, :] for i in range(n)]
L_e=[0.10,0.15,0.20,0.25,0.30]
def T(epsilon):
  v=np.array([1,0,0,1-epsilon])
  V = v.reshape(2,2)
  z=transformer(Time, dt, n, d, A, V, x0)
  return(time_bifurcation(clusters,z,Time,dt,delta)[0],time_bifurcation(clusters,z,Time,dt,delta)[1])

with Pool(5) as p:
  L=p.map(T,L_e)
  for i in range (len(L)):
    L_clus.append(L[i][0])
    L_phase.append(L[i][1])

plt.figure()
plt.plot(L_e,L_clus,'o',label='Time cluster')
plt.plot(L_e,L_phase,'o',label='Time bifurcation')
plt.ylabel('Time bifurcation')
plt.xlabel('epsilon')
plt.xscale('log')
plt.yscale('log')
plt.savefig('time_bifurcation.pdf', format='pdf', bbox_inches='tight')
plt.legend()
plt.show()
