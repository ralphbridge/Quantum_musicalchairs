#include<cuda_runtime.h>
#include<stdio.h>
#include<time.h>
#include<curand.h>
#include<curand_kernel.h>
#include<math.h>
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<algorithm>

#define TPB 256

/*
Euler:  31 4-Byte registers, 24 Bytes of shared memory per thread. 1080Ti => 100.0% occupancy, 57344 pa>

********************************************************************************
*************    THIS VERSION IS NOT OPTIMIZED    ******************************
********************************************************************************
*/

#define N 1000 // Number of electrons

#define steps 300000 // Maximum alloed number of steps to kill simulation

__device__ double dev_traj[6*steps*N]; // Record single paths (both positions and velocities)

__constant__ double pi;
__constant__ double q; // electron charge
__constant__ double m; // electron rest mass
__constant__ double hbar; // Planck's constant
__constant__ double c; // velocity of light in vacuum
__constant__ double eps0;
__constant__ double v0; // electron velocity in the z direction
__constant__ double sigma; // electron beam standard deviation
__constant__ double sigma_p; // electron beam transverse momentum standard deviation

__constant__ double zdet; // Detector position

__constant__ double rmin;
__constant__ double rmax;

__constant__ double dt; // time step for the electron trajectory

void onHost(); // Main CPU function
void onDevice(double *r,double *theta,double *phi); // Main GPU func>

__global__ void setup_randoms(curandState *state, unsigned long seed) {}
int main() { return 0; }
