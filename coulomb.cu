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

#define N 3 // Number of electrons

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
void onDevice(double *r,double *theta,double *phi,double *p,double *theta_p,double *phi_p,double *E_h); // Main GPU function

__global__ void setup_rnd(curandState *state,unsigned long seed); // Sets up seeds for the random number generation 
__global__ void rndvecs(double *x,curandState *state,int option,int n);
__global__ void paths_euler(double *k,double *angles,double *pos);
__global__ void positions(double *vec,double *r,double *theta,double *phi,int opt);
__global__ void Efield(double *pos,double *E);

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

	FILE *x_vec,*E_vec;

	time_t rawtime;
	struct tm*timeinfo;

	time(&rawtime);
	timeinfo=localtime(&rawtime);

	printf("The current time is %s",asctime(timeinfo));
	
	const char* name_x1="initialpositions";
	const char* name_x2="Efield";
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
	double *E_h; // Electric field (just for debugging)
	/*double *v_init_h; // Initial transverse velocities, vector of size 3N
	double *detector_h; // Single vector for the final positions, initial transverse velocities and final positions (6N in length for optimization purposes)*/

	r_h=(double*)malloc(N*sizeof(double));
	theta_h=(double*)malloc(N*sizeof(double));
	phi_h=(double*)malloc(N*sizeof(double));
	
	p_h=(double*)malloc(N*sizeof(double));
	theta_p_h=(double*)malloc(N*sizeof(double));
	phi_p_h=(double*)malloc(N*sizeof(double));
	
	E_h=(double*)malloc(3*N*sizeof(double));

	onDevice(r_h,theta_h,phi_h,p_h,theta_p_h,phi_p_h,E_h); // GPU function that computes the randomly generated positions

	x_vec=fopen(filename_x1,"w");
	for(int i=0;i<N;i++){
		fprintf(x_vec,"%2.8e,%f,%f,%2.8e,%f,%f\n",r_h[i],theta_h[i],phi_h[i],p_h[i],theta_p_h[i],phi_p_h[i]);
	}
	fclose(x_vec);
	
	E_vec=fopen(filename_x2,"w");
	for(int i=0;i<3*N;i=i+3){
		fprintf(E_vec,"%2.8e,%2.8e,%2.8e\n",E_h[i],E_h[i+1],E_h[i+2]);
	}
	fclose(E_vec);

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
	free(E_h);
}

void onDevice(double *r_h,double *theta_h,double *phi_h,double *p_h,double *theta_p_h,double *phi_p_h,double *E_h){
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

	double dt_h=zdet_h/(1000*v0_h); // Think about time step

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
	double *E;

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
	cudaMalloc((void**)&E,3*N*sizeof(double));
	
	cudaMalloc((void**)&E,3*N*sizeof(double));

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
	srand(time(NULL)); // <---- check if this is necessary
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
	
	positions<<<blocks,TPB>>>(r,r_d,theta_d,phi_d,1); // Building cartesian position vector (3N in size) out of GPU-located r,theta and phi vectors
	
	positions<<<blocks,TPB>>>(p,p_d,theta_p_d,phi_p_d,2); // Building cartesian momenta vector (3N in size) out of GPU-located p,theta_p and phi_p vectors
	
	//E field (for debugging only)
	Efield<<<blocks,TPB>>>(r,E);
	
	cudaMemcpy(E_h,E,3*N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(devStates_r);
	cudaFree(r_d);
	cudaFree(theta_d);
	cudaFree(phi_d);
	cudaFree(devStates_p);
	cudaFree(p_d);
	cudaFree(theta_p_d);
	cudaFree(phi_p_d);
	cudaFree(r);
	cudaFree(p);
	cudaFree(E);
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

__global__ void positions(double *vec,double *r,double *theta,double *phi,int opt){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx<N){
		if(opt==1){
			vec[3*idx]=r[idx]*sin(theta[idx])*cos(phi[idx]);
			vec[3*idx+1]=r[idx]*sin(theta[idx])*sin(phi[idx]);
			vec[3*idx+2]=r[idx]*cos(theta[idx]);
		}else{
			vec[3*idx]=r[idx]*sin(theta[idx])*cos(phi[idx]);
			vec[3*idx+1]=r[idx]*sin(theta[idx])*sin(phi[idx]);
			vec[3*idx+2]=r[idx]*cos(theta[idx]);
		}
	}
}

__global__ void Efield(double *pos,double *E){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx<N){
		for(int i=0;i<N;i++){
			if(i!=idx){
				E[3*idx]=pos[i]/pow(pow(pos[i],2.0)+pow(i+1,2.0)+pow(i+2,2.0),3.0/2.0);
				E[3*idx+1]=pos[i+1]/pow(pow(pos[i],2.0)+pow(i+1,2.0)+pow(i+2,2.0),3.0/2.0);
				E[3*idx+2]=pos[i+2]/pow(pow(pos[i],2.0)+pow(i+1,2.0)+pow(i+2,2.0),3.0/2.0);
			}
		}
	}
}

__global__ void paths_euler(double *k,double *angles,double *pos){
	unsigned int idx=threadIdx.x+blockIdx.x*TPB;
	
	__shared__ double vxnn[TPB];
	__shared__ double vynn[TPB];
	__shared__ double vznn[TPB];

	if(idx<N){
		double tn=0.0;
		double xn=0.0;
		double yn=pos[idx];
		double zn=0.0;

		double vxn=0.0;
		double vyn=pos[N+idx];
		__syncthreads();
		double vzn=v0;

		if(coq!=0){
			my_push_back(xn,yn,zn,vxn,vyn,vzn,idx);
		}

		while(zn<=D){
			vxnn[threadIdx.x]=0.0;
			vynn[threadIdx.x]=0.0;
			vznn[threadIdx.x]=0.0;

			for(int i=0;i<Nk;i++){
				__syncthreads();
				vxnn[threadIdx.x]=vxnn[threadIdx.x]+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],angles[3*Nk+i],xi[i],tn,xn,yn,zn,vyn,vzn); // vxnn represents here the total ZPF force in x (recycled variable)
				__syncthreads();
				vynn[threadIdx.x]=vynn[threadIdx.x]+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],angles[3*Nk+i],xi[i],tn,xn,yn,zn,vxn,vzn); // k1vy represents here the total ZPF force in y
				
				__syncthreads();
				vznn[threadIdx.x]=vznn[threadIdx.x]+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],angles[3*Nk+i],xi[i],tn,xn,yn,zn,vxn,vyn); // k1vz represents here the total ZPF force in z
			}
			vynn[threadIdx.x]=vynn[threadIdx.x]+gL(tn,yn,zn,vzn);
			vznn[threadIdx.x]=vznn[threadIdx.x]+hL(tn,yn,zn,vyn);

			__syncthreads();
			vxnn[threadIdx.x]=vxn+dt*vxnn[threadIdx.x];

			__syncthreads();
			vynn[threadIdx.x]=vyn+dt*vynn[threadIdx.x];

			__syncthreads();
			vznn[threadIdx.x]=vzn+dt*vznn[threadIdx.x];

			__syncthreads();
			tn=tn+dt;

			__syncthreads();
			xn=xn+dt*vxn;

			__syncthreads();
			yn=yn+dt*vyn;

			__syncthreads();
			zn=zn+dt*vzn;
		
			vxn=vxnn[threadIdx.x];
			vyn=vynn[threadIdx.x];
			vzn=vznn[threadIdx.x];

			if(coq!=0){
				my_push_back(xn,yn,zn,vxn,vyn,vzn,idx);
			}

		}
		__syncthreads();
		pos[2*N+idx]=yn+(zimp-D)*vyn/vzn;
	}
}
