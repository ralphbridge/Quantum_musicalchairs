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

#define steps 1000 // Maximum allowed number of steps to kill simulation

__device__ double dev_traj[7*2*N]; // Record single paths (both positions and velocities)

__constant__ double pi;
__constant__ double q; // electron charge
__constant__ double m; // electron rest mass
__constant__ double hbar; // Planck's constant
__constant__ double c; // velocity of light in vacuum
__constant__ double k; // 1/(4*pi*epsilon0)
__constant__ double v0; // electron velocity in the z direction
__constant__ double sigma; // electron beam standard deviation
__constant__ double sigma_p; // electron beam transverse momentum standard deviation
__constant__ double sigma_theta_p;

__constant__ double Vtip; // Tip voltage
__constant__ double rtip; // Tip radius of curvature
__constant__ double zdet; // Detector position

__constant__ double rmin; // Minimum spherical shell radius
__constant__ double rmax; // Maximum spherical shell radius

__constant__ double dt; // time step for the electron trajectory

void onHost(); // Main CPU function
void onDevice(double *r,double *theta,double *phi,double *p,double *theta_p,double *phi_p,double *E_h); // Main GPU function

__global__ void setup_rnd(curandState *state,unsigned long seed); // Sets up seeds for the random number generation 
__global__ void rndvecs(double *x,curandState *state,int option,int n);
__global__ void sph2cart(double *vec,double *r,double *theta,double *phi,int n);
__global__ void Efield(double *pos,double *E);
__global__ void Pauli_blockade(double *pos,double *E, double *r_init, double *r_new, double *theta_new, double *phi_new);
__global__ void paths_euler(double *r,double *p,double *E);

__device__ unsigned int dev_count[N]; // Global index that counts (per thread) iteration steps

__device__ void my_push_back(double const &x,double const &y,double const &z,double const &vx,double const &vy,double const &vz,int const &idx){ // Function that loads positions and velocities into device memory per thread, I don't know why I put the variables as constants
	if(dev_count[idx]<steps){
		dev_traj[7*steps*idx+7*dev_count[idx]]=x;
		dev_traj[7*steps*idx+7*dev_count[idx]+1]=y;
		dev_traj[7*steps*idx+7*dev_count[idx]+2]=z;
		dev_traj[7*steps*idx+7*dev_count[idx]+3]=vx;
		dev_traj[7*steps*idx+7*dev_count[idx]+4]=vy;
		dev_traj[7*steps*idx+7*dev_count[idx]+5]=vz;
		dev_traj[7*steps*idx+7*dev_count[idx]+6]=idx;
		dev_count[idx]=dev_count[idx]+1;
	}else{
		printf("Overflow error in pushback\n");
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

	time_t t=time(0);   // get time now
	struct tm *now=localtime(&t);
	char x_vec[80],E_vec[80];
	strftime (x_vec,80,"initialconditions%b%d_%H_%M.txt",now);
	strftime (E_vec,80,"Efield%b%d_%H_%M.txt",now);

	std::cout.precision(15);
	std::ofstream myfile;

	time_t rawtime;
	struct tm*timeinfo;

	time(&rawtime);
	timeinfo=localtime(&rawtime);

	printf("The current time is %s",asctime(timeinfo));

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

	myfile.open(x_vec);
	if(myfile.is_open()){
		for(unsigned i=0;i<N;i++){
			myfile << std::scientific << r_h[i] << ',' << theta_h[i] << ',' << phi_h[i]  << ',' << p_h[i]  << ',' << theta_p_h[i]  << ',' << phi_p_h[i] << '\n';
		}
		std::cout << '\n';
		myfile.close();
	}

	myfile.open(E_vec);
	if(myfile.is_open()){
		for(unsigned i=0;i<3*N;i=i+3){
			myfile << std::scientific << E_h[i] << ',' << E_h[i+1] << ',' << E_h[i+2] << '\n';
		}
		std::cout << '\n';
		myfile.close();
	}

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
	double q_h=-1.6e-19;
	double m_h=9.10938356e-31;
	double hbar_h=1.0545718e-34;
	double c_h=299792458.0;
	double eps0_h=8.85e-12;
	double k_h=1/(4*pi_h*eps0_h);
	double v0_h=1.1e7;

	//double sigma_p_h=0.05*m_h*v0_h;
	double sigma_p_h=5.4e-25; // Arjun suggested to use 1eV uniform distribution for p
	double sigma_theta_p_h=0.01;

	double Vtip_h=100; // Tip voltage
	double rtip_h=100e-9; // Tip radius of curvature
	double zdet_h=10e-2; // Detector position

	double rmin_h=0.0;
	double rmax_h=1e-6;

	double dt_h=zdet_h/(10*v0_h); // Think about time step

	cudaMemcpyToSymbol(pi,&pi_h,sizeof(double)); // Copy parameters to constant memory for optimization purposes
	cudaMemcpyToSymbol(q,&q_h,sizeof(double));
	cudaMemcpyToSymbol(m,&m_h,sizeof(double));
	cudaMemcpyToSymbol(hbar,&hbar_h,sizeof(double));
	cudaMemcpyToSymbol(c,&c_h,sizeof(double));
	cudaMemcpyToSymbol(k,&k_h,sizeof(double));
	cudaMemcpyToSymbol(v0,&v0_h,sizeof(double));

	cudaMemcpyToSymbol(sigma_p,&sigma_p_h,sizeof(double));
	cudaMemcpyToSymbol(sigma_theta_p,&sigma_theta_p_h,sizeof(double));

	cudaMemcpyToSymbol(Vtip,&Vtip_h,sizeof(double));
	cudaMemcpyToSymbol(rtip,&rtip_h,sizeof(double));
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
	
	sph2cart<<<blocks,TPB>>>(r,r_d,theta_d,phi_d,1); // Building cartesian position vector (3N in size) out of GPU-located r,theta and phi vectors
	
	sph2cart<<<blocks,TPB>>>(p,p_d,theta_p_d,phi_p_d,0); // Building cartesian momenta vector (3N in size) out of GPU-located p,theta_p and phi_p vectors
	
	Efield<<<blocks,TPB>>>(r,E);
	
	//E field GPU to CPU migration(for debugging only)
	cudaMemcpy(E_h,E,3*N*sizeof(double),cudaMemcpyDeviceToHost);

	paths_euler<<<blocks,TPB>>>(r,p,E);

	int dsizes=6*steps*N;

	std::vector<double> results(dsizes);
	cudaMemcpyFromSymbol(&(results[0]),dev_traj,dsizes*sizeof(double));

	time_t t=time(0);   // get time now
	struct tm *now=localtime(&t);
	char filename_t[80];
	strftime (filename_t,80,"trajectories%b%d_%H_%M.txt",now);

	std::cout.precision(15);
	std::ofstream myfile;
	myfile.open(filename_t);

	if(myfile.is_open()){
		for(unsigned i=0;i<results.size()-1;i=i+7){
			if(results[i]+results[i+1]!=0){
				myfile << std::scientific << results[i] << ',' << results[i+1] << ',' << results[i+2]  << ',' << results[i+3]  << ',' << results[i+4]  << ',' << results[i+5] << ',' << std::defaultfloat << static_cast<int>(results[i+6]) << '\n';
			}
		}
		std::cout << '\n';
		myfile.close();
	}

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
	cudaFree(dev_traj);
	cudaFree(dev_count);
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
			//vec[idx]=sigma_p*curand_normal(&localState); // Think about initial energy in the z direction
			vec[idx]=sigma_p*curand_uniform(&localState); // Arjun said that he doesn't see why p should have any preference between 0 and 1eV
		}else if(opt==5){ // Random momentum polar angles
			//vec[idx]=sigma_theta_p*curand_normal(&localState);
			vec[idx]=pi*curand_uniform(&localState); // See comment two lines above
		}else if(opt==6){ // Random momentum azimuthal angles
			vec[idx]=2.0*pi*curand_uniform(&localState);
		}
		globalState[idx]=localState; // Update current seed state
	}
}

__global__ void sph2cart(double *vec,double *r,double *theta,double *phi,int opt){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx<N){
		vec[3*idx]=r[idx]*sin(theta[idx])*cos(phi[idx]);
		vec[3*idx+1]=r[idx]*sin(theta[idx])*sin(phi[idx]);
		if(opt==1){ // z coordinate adds constant offset to set origin of coordinates at the tip position
			__syncthreads();
			vec[3*idx+2]=rtip+rmax+r[idx]*cos(theta[idx]);
		}else{
			vec[3*idx+2]=r[idx]*cos(theta[idx]);
		}
	}
}

__global__ void Efield(double *pos,double *E){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;

	//double R1,R2;

	E[3*idx]=0;
	E[3*idx+1]=0;
	E[3*idx+2]=0;
	if(idx<N){
		/*for(int i=0;i<N;i++){ # Comment/uncomment this for cycle to disable/enable the Coulomb repulsion between charges, as well as in line 483
			if(i!=idx){
				__syncthreads();
				E[3*idx]=E[3*idx]+k*q*(pos[3*idx]-pos[3*i])/pow(pow(pos[3*idx]-pos[3*i],2.0)+pow(pos[3*idx+1]-pos[3*i+1],2.0)+pow(pos[3*idx+2]-pos[3*i+2],2.0),3.0/2.0);
				__syncthreads();
				E[3*idx+1]=E[3*idx+1]+k*q*(pos[3*idx+1]-pos[3*i+1])/pow(pow(pos[3*idx]-pos[3*i],2.0)+pow(pos[3*idx+1]-pos[3*i+1],2.0)+pow(pos[3*idx+2]-pos[3*i+2],2.0),3.0/2.0);
				__syncthreads();
				E[3*idx+2]=E[3*idx+2]+k*q*(pos[3*idx+2]-pos[3*i+2])/pow(pow(pos[3*idx]-pos[3*i],2.0)+pow(pos[3*idx+1]-pos[3*i+1],2.0)+pow(pos[3*idx+2]-pos[3*i+2],2.0),3.0/2.0);
			}
		}*/
		__syncthreads();
		E[3*idx]=rtip*Vtip*pos[3*idx]/pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2],2.0),3.0/2.0);
		__syncthreads();
		E[3*idx+1]=rtip*Vtip*pos[3*idx+1]/pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2],2.0),3.0/2.0);
		__syncthreads();
		E[3*idx+2]=rtip*Vtip*pos[3*idx+2]/pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2],2.0),3.0/2.0);
		/*R1=pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2],2.0),1.0/2.0);
		__syncthreads();
		R2=pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2]-2.0*zdet,2.0),1.0/2.0);
		__syncthreads();
		E[3*idx]=E[3*idx]+Vtip*pos[3*idx]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));
		__syncthreads();
		E[3*idx+1]=E[3*idx+1]+Vtip*pos[3*idx+1]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));
		__syncthreads();
		E[3*idx+2]=E[3*idx+2]+Vtip*((pos[3*idx+2]-2.0*zdet)/pow(R1,3.0)-pos[3*idx+2]/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));*/
	}
}

/*
__global__ void Pauli_blockade(double *pos,double *E, double *r_init, double *r_new, double *theta_new, double *phi_new){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	curandState localState=globalState[idx];
	double r_coh=5.0;
	if(idx<N){
		for(int i=0;i<idx;i++){
			r_init[idx]=pow(pow(pos[3*idx]-pos[3*i],2.0)+pow(pos[3*idx+1]-pos[3*i+1],2.0)+pow(pos[3*idx+2]-pos[3*i+2],2.0),1/2.0);
			if(r_init < r_coh){
				r_new[idx]=(r_coh)*curand_uniform(&localState);
				theta_new[idx]=acos(1.0-2.0*curand_uniform(&localState));
				phi_new[idx]=2.0 * pi * curand_uniform(&localState); 
				pos[3*i] = (r_init[idx]+r_new[idx])*sin(theta_new[idx])*cos(phi_new[idx]);
				__syncthreads();
				pos[3*i+1] = (r_init[idx]+r_new[idx])*sin(theta_new[idx])*sin(phi_new[idx]);
				__syncthreads();
				pos[3*i+2] = (r_init[idx]+r_new[idx])*cos(theta_new[idx]);
				__syncthreads();
			}
		}
	}
}
*/

//__global__ void paths_euler(double *k,double *angles,double *pos){
__global__ void paths_euler(double *r,double *p,double *E){
	unsigned int idx=threadIdx.x+blockIdx.x*TPB;

	unsigned int iter=0;
	
	__shared__ double vxnn[TPB];
	__shared__ double vynn[TPB];
	__shared__ double vznn[TPB];

	if(idx<N){
		double tn=0.0;

		__syncthreads();
		double vxn=p[3*idx]/m;
		__syncthreads();
		double vyn=p[3*idx+1]/m;
		__syncthreads();
		double vzn=p[3*idx+2]/m;

		//double R1,R2;

		/*printf("vx=%f for particle %d\n",vxn,idx);
		printf("vy=%f for particle %d\n",vyn,idx);
		printf("vz=%f for particle %d\n",vzn,idx);*/

		if(tn==0){
			my_push_back(r[3*idx],r[3*idx+1],r[3*idx+2],vxn,vyn,vzn,idx);
		}

		while(r[3*idx+2]<=zdet && iter<steps){
			__syncthreads();
			vxnn[threadIdx.x]=vxn+dt*q*E[3*idx]/m; // minus sign to account for the e charge
			__syncthreads();
			vynn[threadIdx.x]=vyn+dt*q*E[3*idx+1]/m;
			__syncthreads();
			vznn[threadIdx.x]=vzn+dt*q*E[3*idx+2]/m;

			__syncthreads();
			tn=tn+dt;

			__syncthreads();
			r[3*idx]=r[3*idx]+dt*vxn;

			__syncthreads();
			r[3*idx+1]=r[3*idx+1]+dt*vyn;

			__syncthreads();
			r[3*idx+2]=r[3*idx+2]+dt*vzn;
		
			vxn=vxnn[threadIdx.x];
			vyn=vynn[threadIdx.x];
			vzn=vznn[threadIdx.x];

			__syncthreads();

			/*for(int i=0;i<N;i++){ # Comment/uncomment this for cycle to disable/enable the Coulomb repulsion between charges, as well as in line 375
				if(i!=idx && r[3*i+2]<zdet){
					E[3*idx]=E[3*idx]+k*q*(r[3*idx]-r[3*i])/pow(pow(r[3*idx]-r[3*i],2.0)+pow(r[3*idx+1]-r[3*i+1],2.0)+pow(r[3*idx+2]-r[3*i+2],2.0),3.0/2.0);
					__syncthreads();
					E[3*idx+1]=E[3*idx+1]+k*q*(r[3*idx+1]-r[3*i+1])/pow(pow(r[3*idx]-r[3*i],2.0)+pow(r[3*idx+1]-r[3*i+1],2.0)+pow(r[3*idx+2]-r[3*i+2],2.0),3.0/2.0);
					__syncthreads();
					E[3*idx+2]=E[3*idx+2]+k*q*(r[3*idx+2]-r[3*i+2])/pow(pow(r[3*idx]-r[3*i],2.0)+pow(r[3*idx+1]-r[3*i+1],2.0)+pow(r[3*idx+2]-r[3*i+2],2.0),3.0/2.0);
				}
			}*/
			__syncthreads();
			E[3*idx]=rtip*Vtip*r[3*idx]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0);
			__syncthreads();
			E[3*idx+1]=rtip*Vtip*r[3*idx+1]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0);
			__syncthreads();
			E[3*idx+2]=rtip*Vtip*r[3*idx+2]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0);
			/*R1=pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),1.0/2.0);
			__syncthreads();
			R2=pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2]-2.0*zdet,2.0),1.0/2.0);
			__syncthreads();
			E[3*idx]=E[3*idx]+Vtip*r[3*idx]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));
			__syncthreads();
			E[3*idx+1]=E[3*idx+1]+Vtip*r[3*idx+1]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));
			__syncthreads();
			E[3*idx+2]=E[3*idx+2]+Vtip*((r[3*idx+2]-2.0*zdet)/pow(R1,3.0)-r[3*idx+2]/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));*/

			if(iter==20){
				printf("x=%f for particle %d\n",r[3*idx],idx);
				printf("y=%f for particle %d\n",r[3*idx+1],idx);
				printf("z=%f for particle %d\n",r[3*idx+2],idx);
				//printf("R1=%f for particle %d\n",R1,idx);
				//printf("R2=%f for particle %d\n",R2,idx);
				__syncthreads();
				//printf("Ex=%f for particle %d\n",Vtip*r[3*idx]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet)),idx);
				printf("Ex=%f for particle %d\n",rtip*Vtip*r[3*idx]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0),idx);
				__syncthreads();
				//printf("Ey=%f for particle %d\n",Vtip*r[3*idx+1]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet)),idx);
				printf("Ey=%f for particle %d\n",rtip*Vtip*r[3*idx+1]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0),idx);
				__syncthreads();
				//printf("Ez=%f for particle %d\n",Vtip*((r[3*idx+2]-2.0*zdet)/pow(R1,3.0)-r[3*idx+2]/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet)),idx);
				printf("Ez=%f for particle %d\n",rtip*Vtip*r[3*idx+2]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0),idx);
			}

			++iter;
			if(r[3*idx+2]>=zdet || iter==steps){
				my_push_back(r[3*idx],r[3*idx+1],r[3*idx+2],vxn,vyn,vzn,idx);
				if(iter==steps){
					printf("Particle %d did not make it to the detector\n",idx);
				}
			}
		}
		//__syncthreads();
	}
}
