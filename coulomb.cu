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
Euler:	31 4-Byte registers, 24 Bytes of shared memory per thread. 1080Ti => 100.0% occupancy, 57344 particles simultaneously.

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
__constant__ double sigma_theta_p;

__constant__ double zdet; // Detector position

__constant__ double rmin;
__constant__ double rmax;

__constant__ double dt; // time step for the electron trajectory

void onHost(); // Main CPU function
void onDevice(double *r,double *theta,double *phi,double *p,double *theta_p,double *phi_p); // Main GPU function

__global__ void setup_rnd(curandState *state,unsigned long seed); // Sets up seeds for the random number generation 
__global__ void rndvecs(double *x,curandState *state,int option,int n);
__global__ void paths_euler(double *k,double *angles,double *pos);

__device__ unsigned int dev_count[N]; // Global index that counts (per thread) iteration steps

__device__ void my_push_back(double const &x,double const &y,double const &z,double const &vx,double const &vy,double const &vz,int const &idx){ // Function that loads positions and velocities into device memory per thread, I don't know why I put the variables as constants
	if(dev_count[idx]<steps){
		dev_traj[6*steps*idx+6*dev_count[idx]]=x;
		dev_traj[6*steps*idx+6*dev_count[idx]+1]=y;
		dev_traj[6*steps*idx+6*dev_count[idx]+2]=z;
		dev_traj[6*steps*idx+6*dev_count[idx]+3]=vx;
		dev_traj[6*steps*idx+6*dev_count[idx]+4]=vy;
		dev_traj[6*steps*idx+6*dev_count[idx]+5]=vz;	
		dev_count[idx]=dev_count[idx]+1;
	}else{
		printf("Overflow error (in pushback)\n");
	}
}

int main(){
	onHost();
	return 0;
}

void onHost(){

	// This section sets up the timer to measure execution times and filenames
	float elapsedTime;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	FILE *x_vec; //,*posit=NULL;

	time_t rawtime;
	struct tm*timeinfo;

	time(&rawtime);
	timeinfo=localtime(&rawtime);

	printf("The current time is %s",asctime(timeinfo));
	
	const char* name_x1="initialpositions";
	const char* name_x2="positions";
	const char* format=".txt";

	char day[10];

	strftime(day, sizeof(day)-1, "%d_%H_%M", timeinfo);

	char strtmp[6];

	char filename_x1[512];
	char filename_x2[512];

	std::copy(asctime(timeinfo)+4,asctime(timeinfo)+7,strtmp);

	sprintf(filename_x1,"%s%s%s%s",name_x1,strtmp,day,format);
	sprintf(filename_x2,"%s%s%s%s",name_x2,strtmp,day,format);

	// This section computes the random initial position and momentum distributions
	double *r_h,*theta_h,*phi_h; // Spherical coordinates for each particle's initial positions (N in total)
	double *p_h,*theta_p_h,*phi_p_h; // Initial momenta in spherical coordinates (N in total)
	/*double *v_init_h; // Initial transverse velocities, vector of size 3N
	double *detector_h; // Single vector for the final positions, initial transverse velocities and final positions (6N in length for optimization purposes)*/

	r_h=(double*)malloc(N*sizeof(double));
	theta_h=(double*)malloc(N*sizeof(double));
	phi_h=(double*)malloc(N*sizeof(double));
	
	p_h=(double*)malloc(N*sizeof(double));
	theta_p_h=(double*)malloc(N*sizeof(double));
	phi_p_h=(double*)malloc(N*sizeof(double));

	onDevice(r_h,theta_h,phi_h,p_h,theta_p_h,phi_p_h); // GPU function that computes the randomly generated positions

	x_vec=fopen(filename_x1,"w");
	for(int i=0;i<N;i++){
		fprintf(x_vec,"%2.8e,%f,%f,%2.8e,%f,%f\n",r_h[i],theta_h[i],phi_h[i],p_h[i],theta_p_h[i],phi_p_h[i]);
	}
	fclose(x_vec);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Total time: %6.4f hours\n",elapsedTime*1e-3/3600.0);
	printf("------------------------------------------------------------\n");

	free(r_h);
	free(theta_h);
	free(phi_h);
	free(p_h);
	free(theta_p_h);
	free(phi_p_h);
}

void onDevice(double *r_h,double *theta_h,double *phi_h,double *p_h,double *theta_p_h,double *phi_p_h){
	unsigned int blocks=(N+TPB-1)/TPB; // Check this line for optimization purposes

	double pi_h=3.1415926535;
	double q_h=1.6e-19;
	double m_h=9.10938356e-31;
	double hbar_h=1.0545718e-34;
	double c_h=299792458.0;
	double eps0_h=8.85e-12;
	double v0_h=1.1e7;

	double sigma_p_h=0.05*m_h*v0_h;
	double sigma_theta_p_h=0.01;

	double zdet_h=10e-2; // Detector position

	double rmin_h=1e-6;
	double rmax_h=0.01e-6;

	double dt_h=zdet_h/(100*v0_h); // Think about time step

	cudaMemcpyToSymbol(pi,&pi_h,sizeof(double)); // Copy parameters to constant memory for optimization purposes
	cudaMemcpyToSymbol(q,&q_h,sizeof(double));
	cudaMemcpyToSymbol(m,&m_h,sizeof(double));
	cudaMemcpyToSymbol(hbar,&hbar_h,sizeof(double));
	cudaMemcpyToSymbol(c,&c_h,sizeof(double));
	cudaMemcpyToSymbol(eps0,&eps0_h,sizeof(double));
	cudaMemcpyToSymbol(v0,&v0_h,sizeof(double));

	cudaMemcpyToSymbol(sigma_p,&sigma_p_h,sizeof(double));
	cudaMemcpyToSymbol(sigma_theta_p,&sigma_theta_p_h,sizeof(double));

	cudaMemcpyToSymbol(zdet,&zdet_h,sizeof(double));

	cudaMemcpyToSymbol(rmin,&rmin_h,sizeof(double));
	cudaMemcpyToSymbol(rmax,&rmax_h,sizeof(double));

	cudaMemcpyToSymbol(dt,&dt_h,sizeof(double));

	double *r_d,*theta_d,*phi_d;
	double *p_d,*theta_p_d,*phi_p_d;
	double *r,*p;

	printf("Coulomb explosion\n");
	printf("Number of particles (N): %d\n",N);
	printf("r_min=%2.6e m\n",rmin_h);
	printf("r_max=%2.6e m\n",rmax_h);

	printf("sigmap=%2.6e kg*m/s\n",sigma_p_h);
	printf("sigmathetap=%f rad\n",sigma_theta_p_h);

	printf("dt=%2.6e s\n",dt_h);
	
	printf("Threads per block: %d\n",TPB);
	printf("Number of blocks: %d\n",blocks);

	cudaMalloc((void**)&r_d,N*sizeof(double));
	cudaMalloc((void**)&theta_d,N*sizeof(double));
	cudaMalloc((void**)&phi_d,N*sizeof(double));
	
	cudaMalloc((void**)&p_d,N*sizeof(double));
	cudaMalloc((void**)&theta_p_d,N*sizeof(double));
	cudaMalloc((void**)&phi_p_d,N*sizeof(double));
	
	cudaMalloc((void**)&r,3*N*sizeof(double));
	cudaMalloc((void**)&p,3*N*sizeof(double));

	curandState *devStates_r;
	cudaMalloc(&devStates_r,N*sizeof(curandState));

	//r
	srand(time(0));
	int seed=rand(); //Setting up the seeds
	setup_rnd<<<blocks,TPB>>>(devStates_r,seed);

	rndvecs<<<blocks,TPB>>>(r_d,devStates_r,1,N);

	//theta
	rndvecs<<<blocks,TPB>>>(theta_d,devStates_r,2,N);

	//phi
	rndvecs<<<blocks,TPB>>>(phi_d,devStates_r,3,N);

	cudaMemcpy(r_h,r_d,N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(theta_h,theta_d,N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(phi_h,phi_d,N*sizeof(double),cudaMemcpyDeviceToHost);
	
	curandState *devStates_p;
	cudaMalloc(&devStates_p,N*sizeof(curandState));
	
	//p
	srand(time(NULL));
	seed=rand(); //Setting up the seeds <---- check if this is necessary
	setup_rnd<<<blocks,TPB>>>(devStates_p,seed);

	rndvecs<<<blocks,TPB>>>(p_d,devStates_p,4,N);

	//theta_p
	rndvecs<<<blocks,TPB>>>(theta_p_d,devStates_p,5,N);

	//phi_p
	rndvecs<<<blocks,TPB>>>(phi_p_d,devStates_p,6,N);

	cudaMemcpy(p_h,p_d,N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(theta_p_h,theta_p_d,N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(phi_p_h,phi_p_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(devStates_r);
	cudaFree(r_d);
	cudaFree(theta_d);
	cudaFree(phi_d);
	cudaFree(devStates_p);
	cudaFree(p_d);
	cudaFree(theta_p_d);
	cudaFree(phi_p_d);
}

__global__ void setup_rnd(curandState *state,unsigned long seed){
        int idx=threadIdx.x+blockIdx.x*blockDim.x;
        curand_init(seed,idx,0,&state[idx]); // Initializes the random state
}

__global__ void rndvecs(double *vec,curandState *globalState,int opt,int n){ // Random number generation
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	curandState localState=globalState[idx];
	if(idx<n){
		if(opt==1){ // Random radii
			vec[idx]=pow((pow(rmax,3.0)-pow(rmin,3.0))*curand_uniform(&localState)+pow(rmin,3.0),1.0/3.0);
		}else if(opt==2){ // Random polar angles
			vec[idx]=acos(1.0-2.0*curand_uniform(&localState));
		}else if(opt==3){ // Random azimuthal angles
			vec[idx]=2.0*pi*curand_uniform(&localState);
		}else if(opt==4){ // Random momenta magnitude
			vec[idx]=sigma_p*curand_normal(&localState)+m*v0;
		}else if(opt==5){ // Random momentum polar angles
			vec[idx]=sigma_theta_p*curand_normal(&localState);
		}else if(opt==6){ // Random momentum azimuthal angles
			vec[idx]=2.0*pi*curand_uniform(&localState);
		}
		globalState[idx]=localState; // Update current seed state
	}
}
