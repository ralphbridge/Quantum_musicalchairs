import numpy as np
from scipy import constants as const
from scipy.spatial import distance
#import cupy as cp
import random as rnd
import math

def for_method(pos,q):
    """Computes electric field vectors for all particles using for-loop."""
    Evect = np.zeros( (len(pos),len(pos[0])) ) # define output electric field vector
    k =  1 / (4 * np.pi * const.epsilon_0) * np.ones((len(pos),len(pos[0]))) * const.e # make this into matrix as matrix addition is faster
    # alternatively you can get rid of np.ones and just define this as a number
    
    for i, v0 in enumerate(pos): # s_p - selected particle | iterate over all particles | v0 reference particle
        for v, qc in zip(pos,q): # loop over all particles and calculate electric force sum | v particle being calculated for
            if all((v0 == v)):   # do not compute for the same particle
                continue
            else:
                r = v0 - v       #
                Evect[i] += r / np.linalg.norm(r) ** 3 * qc #! multiply by charge
    return Evect * k

def CPU_matrix_method(pos,q):
    """Classic vectorization of for Coulomb law using numpy arrays."""
    k = 1 / (4 * np.pi * const.epsilon_0) * np.ones((len(pos),3)) * const.e # define electric constant
    dist = distance.cdist(pos,pos)  # compute distances
    return k * np.sum( (( np.tile(pos,len(pos)).reshape((len(pos),len(pos),3)) - np.tile(pos,(len(pos),1,1))) * q.reshape(len(q),1)).T * np.power(dist,-3, where = dist != 0),axis = 1).T
"""
def GPU_matrix_method(pos,q):
    #GPU Coulomb law vectorization. Takes in numpy arrays, performs computations and returns cupy array
    # compute distance matrix between each particle
    k_cp = 1 / (4 * cp.pi * const.epsilon_0) * cp.ones((len(pos),3)) * const.e # define electric constant, runs faster if this is matrix
    dist = cp.array(distance.cdist(pos,pos)) # could speed this up with cupy cdist function! use this: cupyx.scipy.spatial.distance.cdist
    pos, q = cp.array(pos), cp.array(q) # load inputs to GPU memory
    dist_mod = cp.power(dist,-3)        # compute inverse cube of distance
    dist_mod[dist_mod == cp.inf] = 0    # set all infinity entries to 0 (i.e. diagonal elements/ same particle-particle pairs)
    # compute by magic
    return k_cp * cp.sum((( cp.tile(pos,len(pos)).reshape((len(pos),len(pos),3)) - cp.tile(pos,(len(pos),1,1))) * q.reshape(len(q),1)).T * dist_mod, axis = 1).T
"""

N=3 # Number of partices
rmin=10e-9 # Minumum radius for the positions vector
rmax=100e-9 # Maximum radius for the positions vector
v=1.1e7 # Nominal electron velocity
sigma_p=0.05*const.m_e*v


particles = np.array([[1,0,0],[2,1,0],[2,2,0],[1,1,1],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[-1,0,0],[-1,0,1],[-1,-1,0],[-1,0,1],[-1,0,-1],[-1,-1,-1]]) # location of each particle
q = np.array([1,1,1,-1,2,-2,1.5,-1.5,0.5,-0.5,-0.3,-1,0.32,0.98]) # charge of each particle

E_for=for_method(particles,q)
E_cpu=CPU_matrix_method(particles,q)
#E_gpu=GPU_matrix_method(particles,q)

#print(E_for)
#print(E_cpu)
#print(E_gpu)

# This section computes the radius, polar angle and azimuthal angle for the posiions of the N particles in the random shell of radius rmax-rmin

r=np.zeros(N)
theta=np.zeros(N)
phi=np.zeros(N)

for i in range(N):
    r[i] = ((rmax**3.0-rmin**3.0)*rnd.random()+rmin**3.0)**(1/3) # pow((pow(kmax,3.0)-pow(kmin,3.0))*curand_uniform(&localState)+pow(kmin,3.0),1.0/3.0)
    theta[i] = math.acos(1-2*rnd.radom()) # acos(1.0-2.0*curand_uniform(&localState));
    phi[i] = 2*np.pi*rd.random() # 2.0*pi*curand_uniform(&localState);
    
print(r)
print(theta)
print(phi)

# This section computes the initial momentum distribution for the N particles in a Gaussian distribution (both angular and in magnitude)

p=np.zeros(N)
theta_p=np.zeros(N)
phi_p=np.zeros(N)

for i in range(N):
    p[i]=rnd.gauss(const.m_e*v,sigma_p)
    theta_p[i]=

