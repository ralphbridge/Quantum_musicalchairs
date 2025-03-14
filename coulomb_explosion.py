import numpy as np
from scipy import constants as const
from scipy.spatial import distance
#import cupy as cp # Use this package for CPU-GPU interaction
import random as rnd
import math

# The following function computes the field due to N particles using brute-force CPU instructions
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

# The following function computes the field due to N particles using optimized CPU instructions
def CPU_matrix_method(pos,q):
    """Classic vectorization of for Coulomb law using numpy arrays."""
    k = 1 / (4 * np.pi * const.epsilon_0) * np.ones((len(pos),3)) * const.e # define electric constant
    dist = distance.cdist(pos,pos)  # compute distances
    return k * np.sum( (( np.tile(pos,len(pos)).reshape((len(pos),len(pos),3)) - np.tile(pos,(len(pos),1,1))) * q.reshape(len(q),1)).T * np.power(dist,-3, where = dist != 0),axis = 1).T
"""
# The following function computes the field due to N particles using optimized GPU instructions
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

def paths_euler(yn,Fn): # Need to compute RK4 version of this part
    return yn+Dt*Fn

N = 3 # Number of partices
rmin = 10e-9 # Minumum radius for the positions vector
rmax = 100e-9 # Maximum radius for the positions vector
v = 1.1e7 # Nominal electron velocity
sigma_p = 0.05*const.m_e*v # Spread for the magnitude of the initial momentum
sigma_theta_p = 0.01 # Spread for the polar angle of the initial momentum
Dt = 1e-9 # Time step for the discretized differentials
zdet = 10e-2 # Distance between tip and detectors
q=-1*np.ones(N) # Charge (over the electron charge) of each of the N particles in adimenstional units

# This section computes the radius, polar angle and azimuthal angle for the posiions of the N particles in the random shell of radius rmax-rmin
r = np.zeros(N)
theta = np.zeros(N)
phi = np.zeros(N)

for i in range(N):
    r[i] = ((rmax**3.0-rmin**3.0)*rnd.random()+rmin**3.0)**(1/3) # pow((pow(kmax,3.0)-pow(kmin,3.0))*curand_uniform(&localState)+pow(kmin,3.0),1.0/3.0)
    theta[i] = math.acos(1-2*rnd.random()) # acos(1.0-2.0*curand_uniform(&localState));
    phi[i] = 2*np.pi*rnd.random() # 2.0*pi*curand_uniform(&localState);

"""
print(r)
print(theta)
print(phi)
"""

# This section computes the initial momentum distribution for the N particles in a Gaussian distribution (both angular and in magnitude)
pmag = np.zeros(N)
theta_p = np.zeros(N)
phi_p = np.zeros(N)

for i in range(N):
    pmag[i] = rnd.gauss(const.m_e*v,sigma_p)
    theta_p[i]= rnd.gauss(0,sigma_theta_p) # Beam aiming towards Z direction
    phi_p[i] = 2*np.pi*rnd.random()

"""
print(pmag)
print(theta_p)
print(phi_p)
"""

cart_pos = np.zeros((N,3))
cart_p = np.zeros((N,3))

for i in range(N): # Initial r and p vectors
    cart_pos[i][0] = r[i]*np.sin(theta[i])*np.cos(phi[i])
    cart_pos[i][1] = r[i]*np.sin(theta[i])*np.sin(phi[i])
    cart_pos[i][2] = r[i]*np.cos(theta[i])
    cart_p[i][0] = pmag[i]*np.sin(theta_p[i])*np.cos(phi_p[i])
    cart_p[i][1] = pmag[i]*np.sin(theta_p[i])*np.sin(phi_p[i])
    cart_p[i][2] = pmag[i]*np.cos(theta_p[i]) 

# E = for_method(cart_pos,q) # Used for benchmarks
E = CPU_matrix_method(cart_pos,q)
# E = GPU_matrix_method(cart_pos,q) # USE ONLY if you have an NVIDIA GPU

print(cart_pos[:,2])

it=0
itmax=100000 # Safety iteration index to prevent infinite loops

for i in range(N-1):
    while min(cart_pos[:,2])<zdet and it<itmax:
        cart_p[i+1][0] = paths_euler(cart_p[i][0],q[i]*E[i][0])
        cart_p[i+1][1] = paths_euler(cart_p[i][1],q[i]*E[i][1])
        cart_p[i+1][2] = paths_euler(cart_p[i][2],q[i]*E[i][2])
        cart_pos[i+1][0] = paths_euler(cart_pos[i][0],cart_p[i][0]/const.m_e)
        cart_pos[i+1][1] = paths_euler(cart_pos[i][1],cart_p[i][1]/const.m_e)
        cart_pos[i+1][2] = paths_euler(cart_pos[i][2],cart_p[i][2]/const.m_e)
        E=CPU_matrix_method(cart_pos,q)
        it+=1

print(cart_pos[:,2])