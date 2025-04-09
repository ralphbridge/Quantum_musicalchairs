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

__constant__ double zdet; // Detector position

__constant__ double rmin;
__constant__ double rmax;

__constant__ double dt; // time step for the electron trajectory

void onHost(); // Main CPU function
void onDevice(double *r,double *theta,double *phi)//,double *v_init,double *detector); // Main GPU function

__global__ void setup_randoms(curandState *state,unsigned long seed); // Sets up seeds for the random number generation 
__global__ void positions(double *x,curandState *state,int option,int n);
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

	float elapsedTime; // Variables to record execution times
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	FILE *k_vec,*posit=NULL;

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

	double *r_h,*theta_h,*phi_h; // Spherical coordinates for each particle's initial positions (N in total)
	double *v_init_h; // Initial transverse velocities, vector of size 3N
	double *detector_h; // Single vector for the final positions, initial transverse velocities and final positions (6N in length for optimization purposes)

	r_h=(double*)malloc(N*sizeof(double));
	theta_h=(double*)malloc(N*sizeof(double));
	phi_h=(double*)malloc(N*sizeof(double));

//	v_init_h=(double*)malloc(3*N*sizeof(double));

//	detector_h=(double*)malloc(6*N*sizeof(double));

	onDevice(r_h,theta_h,phi_h)//,v_init_h,detector_h);

	x_vec=fopen(filename_x1,"w");
	for(int i=0;i<Nk;i++){
		fprintf(x_vec,"%2.8e,%f,%f\n",k_h[i],theta_h[i],phi_h[i]);
	}
	fclose(k_vec);

	/*
	posit=fopen(filename_p,"w");
	for(int i=0;i<N;i++){
		fprintf(posit,"%2.6e,%2.6e,%2.6e\n",detector_h[i],detector_h[N+i],detector_h[2*N+i]);
	}
	fclose(posit);
	*/

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Total time: %6.4f hours\n",elapsedTime*1e-3/3600.0);
	printf("------------------------------------------------------------\n");

	free(r_h);
	free(theta_h);
	free(phi_h);
//	free(v_init_h);
//	free(detector_h);
}

void onDevice(double *r_h,double *theta_h,double *phi_h){
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

	double zdet_h=10e-2;

	double rmin_h=1e-6;
	double rmax_h=0.01e-6;

	double dt_h=zdet_h/(100*v0_h); // Think about time step

	cudaMemcpyToSymbol(pi,&pi_h,sizeof(double));
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
//	double *v_init_d;
//	double *detector_d;

	printf("Coulomb explosion\n");
	printf("Number of particles (N): %d\n",N);
	printf("r_min=%2.6e m\n",kmin_h);
	printf("r_max=%2.6e m\n",kmax_h);

	printf("sigmap=%2.6e kg*m/s\n",sigma_p_h);
	printf("sigmathetap=%f rad\n",sigma_theta_p_h);

	printf("dt=%2.6e s\n",dt_h);
	
	printf("Threads per block: %d\n",TPB);
	printf("Number of blocks: %d\n",blocks);

	cudaMalloc((void**)&r_d,N*sizeof(double));
	cudaMalloc((void**)&theta_d,N*sizeof(double));
	cudaMalloc((void**)&phi_d,N*sizeof(double));

/*	cudaMalloc((void**)&v_init_d,N*sizeof(double));

	cudaMalloc((void**)&detector_d,3*N*sizeof(double));*/

	/* Randomly generated positions inside the spherical shell */

	curandState *devStates_r;
        cudaMalloc(&devStates_r,N*sizeof(curandState));

	//r
	srand(time(0));
	int seed=rand(); //Setting up the seeds
	setup_r<<<blocks,TPB>>>(devStates_r,seed);

	rvecs<<<blocks,TPB>>>(k_d,devStates_r,1,N);

	//theta
	rvecs<<<blocks,TPB>>>(theta_d,devStates_r,2,N);

	//phi
	rvecs<<<blocks,TPB>>>(phi_d,devStates_r,3,N);

	cudaMemcpy(k_h,k_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(theta_h,theta_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(phi_h,phi_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);


	/* Initial positions and transverse momentum*/

	curandState *devStates_init;
	cudaMalloc(&devStates_init,N*sizeof(curandState));

	blocks=(N+TPB-1)/TPB;
	printf("Number of blocks (paths): %d\n",blocks);

	srand(time(NULL));
	seed=rand();
	setup_r<<<blocks,TPB>>>(devStates_init,seed);

	rvecs<<<blocks,TPB>>>(init_d,devStates_init,4,N);
	cudaMemcpy(init_h,init_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	rvecs<<<blocks,TPB>>>(v_init_d,devStates_init,5,N);
	cudaMemcpy(v_init_h,v_init_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	/* Making a single vector for the initial and final positions (reduces the size of memory, one double pointer instead of two) */

	for(int i=0;i<N;i++){
		detector_h[i]=init_h[i];
		detector_h[N+i]=v_init_h[i];
		detector_h[2*N+i]=0.0;
	}

	cudaMemcpy(detector_d,detector_h,3*N*sizeof(double),cudaMemcpyHostToDevice);

	int dsize[N];

	paths_euler<<<blocks,TPB>>>(k_d,angles_d,detector_d);

	printf("Paths computed using Euler method in %6.4f hours\n",elapsedTime*1e-3/3600.0);
	
	cudaMemcpy(detector_h,detector_d,3*N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaMemcpyFromSymbol(&dsize,dev_count,N*sizeof(int));

	int dsizes=6*steps*N;

	if(dsizes>6*steps*N){
		printf("Overflow error\n");
		abort();
	}
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
		for(unsigned i=0;i<results.size()-1;i=i+6){
			if(results[i]+results[i+1]!=0){
				myfile << std::scientific << results[i] << ',' << results[i+1] << ',' << results[i+2]  << ',' << results[i+3]  << ',' << results[i+4]  << ',' << results[i+5] << '\n';
			}
		}
		std::cout << '\n';
		myfile.close();
	}

	cudaFree(devStates_r);
	cudaFree(devStates_eta);
	cudaFree(devStates_init);
	cudaFree(k_d);
	cudaFree(theta_d);
	cudaFree(phi_d);
	cudaFree(v_init_d);
	cudaFree(detector_d);
}

__global__ void setup_r(curandState *state,unsigned long seed){
        int idx=threadIdx.x+blockIdx.x*blockDim.x;
        curand_init(seed,idx,0,&state[idx]);
}

__global__ void rvecs(double *vec,curandState *globalState,int opt,int n){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	curandState localState=globalState[idx];
	if(idx<n){
		if(opt==1){ // Random radii
			vec[idx]=pow((pow(kmax,3.0)-pow(kmin,3.0))*curand_uniform(&localState)+pow(kmin,3.0),1.0/3.0);
		}else if(opt==2){ // Random polar angles
			vec[idx]=acos(1.0-2.0*curand_uniform(&localState));
		}else if(opt==3){ // Random azimuthal angles
			vec[idx]=2.0*pi*curand_uniform(&localState);
		}else if(opt==4){
			vec[idx]=sigma*curand_normal(&localState); // Random initial positions
		}else if(opt==5){
			vec[idx]=sigma_p*curand_normal(&localState); // Random initial transverse momentum
		}else if(opt==6){
			vec[idx]=2.0*pi*curand_uniform(&localState);
		}
		globalState[idx]=localState; // Update current seed state
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

__device__ double f(double const &k,double const &theta,double const &phi,double const &eta1,double const &eta2,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vy,double const &vz){ // ZPF, x-component (Wayne-Herman version)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double alpha=k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z);

	__syncthreads();
	double E0=sqrt(hbar*w/(eps0*V));

	__syncthreads();
	return 2*q*E0*cos(alpha)*(cos(theta)*cos(phi)*(cos(w*t+eta1)*cos(xi)-cos(w*t+eta2)*sin(xi))-sin(phi)*(cos(w*t+eta1)*sin(xi)+cos(w*t+eta2)*cos(xi)))+2*q*E0*sin(alpha)*(sin(theta)*(sin(w*t+eta1)*sin(xi)+sin(w*t+eta2)*cos(xi))*vy+cos(theta)*sin(phi)*(sin(w*t+eta1)*sin(xi)+sin(w*t+eta2)*cos(xi))*vz-cos(phi)*(sin(w*t+eta1)*cos(xi)-sin(w*t+eta2)*sin(xi))*vz)/(m*c);
}

__device__ double g(double const &k,double const &theta,double const &phi,double const &eta1,double const &eta2,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vz){ // ZPF, y-component (Wayne-Herman version)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double alpha=k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z);

	__syncthreads();
	double E0=sqrt(hbar*w/(eps0*V));

	__syncthreads();
	return 2*q*E0*cos(alpha)*(cos(theta)*sin(phi)*(cos(w*t+eta1)*cos(xi)-cos(w*t+eta2)*sin(xi))+cos(phi)*(cos(w*t+eta1)*sin(xi)+cos(w*t+eta2)*cos(xi)))-2*q*E0*sin(alpha)*(sin(theta)*(sin(w*t+eta1)*sin(xi)+sin(w*t+eta2)*cos(xi))*vx+cos(theta)*cos(phi)*(sin(w*t+eta1)*sin(xi)+sin(w*t+eta2)*cos(xi))*vz+sin(phi)*(sin(w*t+eta1)*cos(xi)-sin(w*t+eta2)*sin(xi))*vz)/(m*c);
}

__device__ double gL(double const &t,double const &y,double const &z,double const &vz){ // Laser region, y-component
	__syncthreads();
	double phi1=wL*t-kL*y;
	__syncthreads();
	double phi2=wL*t+kL*y;

	__syncthreads();
	double E0=E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	
	__syncthreads();
	return q*E0*(cos(phi1)-cos(phi2))*vz/(m*c);
}

__device__ double h(double const &k,double const &theta,double const &phi,double const &eta1,double const &eta2,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vy){ // ZPF, z-component (Wayne-Herman version)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double alpha=k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z);
	
	__syncthreads();
	double E0=sqrt(hbar*w/(eps0*V));

	__syncthreads();
	return 2*q*E0*cos(alpha)*(sin(theta)*(-cos(w*t+eta1)*cos(xi)+cos(w*t+eta2)*sin(xi)))+2*q*E0*sin(alpha)*(-cos(theta)*sin(phi)*(sin(w*t+eta1)*sin(xi)+sin(w*t+eta2)*cos(xi))*vx+cos(phi)*(sin(w*t+eta1)*cos(xi)-sin(w*t+eta2)*sin(xi))*vx+cos(theta)*cos(phi)*(sin(w*t+eta1)*sin(xi)+sin(w*t+eta2)*cos(xi))*vy+sin(phi)*(sin(w*t+eta1)*cos(xi)-sin(w*t+eta2)*sin(xi))*vy)/(m*c);
}

__device__ double hL(double const &t,double const &y,double const &z,double const &vy){ // Laser region, z-component
	__syncthreads();
	double phi1=wL*t-kL*y;
	__syncthreads();
	double phi2=wL*t+kL*y;

	__syncthreads();
	double E0=E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	
	__syncthreads();
	return q*E0*(cos(phi1)+cos(phi2))/m-q*E0*(cos(phi1)-cos(phi2))*vy/(m*c);
}
