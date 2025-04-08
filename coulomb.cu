#include<cuda_runtime.h>
#include<stdio.h>
#include<time.h>
//#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include"math.h"
#include<iostream>
#include<fstream>
#include<sstream>

#define TPB 256

/*
When using different methods (Euler, RK2 or RK4) there are different memory settings.
Euler:	31 4-Byte registers, 24 Bytes of shared memory per thread. 1080Ti => 100.0% occupancy, 57344 particles simultaneously.
RK2:	37 4-Byte registers, 48 Bytes of shared memory per thread. 1800Ti =>  75.0% occupancy, 43008 particles simultaneously.
RK4:	43 4-Byte registers, 72 Bytes of shared memory per thread. 1080Ti =>  62.5% occupancy, 35840 particles simultaneously.

********************************************************************************
*************    THIS VERSION IS NOT OPTIMIZED    ******************************
********************************************************************************
*/

#define N 10000 // Number of electrons
#define Nk 1 // Number of k-modes
#define Ne 1 // Number of polarizations per k-mode

__constant__ double pi;
__constant__ double q; // electron charge
__constant__ double m; // electron rest mass
__constant__ double hbar; // Planck's constant
__constant__ double c; // velocity of light in vacuum
__constant__ double eps0;
__constant__ double v0; // electron velocity before laser region
__constant__ double sigma; // electron beam standard deviation
__constant__ double sigma_p; // electron beam transverse momentum standard deviation

__constant__ double wL; // Laser frequency
__constant__ double kL; // kL=wL/c
__constant__ double lamL; // lamL=2pi/kL

__constant__ double wR; // ZPF frequency (resonance)
__constant__ double kR; // kR=wR/c
__constant__ double lamR; // lamR=2pi/kR

__constant__ double E0L; // Laser electric field intensity amplitude
__constant__ double D; // Laser beam waist
__constant__ double zimp; // Screen position (origin set right before laser region)
__constant__ double sigmaL; // laser region standard deviation

__constant__ double damping; // Damping rate (harmonic oscillator approximation)
__constant__ double Delta; // thickness of the spherical shell in k-space using resonance frequency
__constant__ double kmin;
__constant__ double kmax;
__constant__ double V; // Estimated total volume of space

__constant__ double dt; // time step necessary to resolve the electron trajectory

__constant__ double xi[Ne]; // Polarization angles for each k-mode (Ne in total): NOT random, allocated in CONSTANT memory for optimization purposes

void onHost();
void onDevice(double *k,double *theta,double *phi,double *eta,double *angles,double *xi,double *init,double *v_init,double *positions);

__global__ void setup_kmodes(curandState *state,unsigned long seed);
__global__ void kmodes(double *x,curandState *state,int option,int n);
__global__ void paths_euler(double *k,double *angles,double *pos);
__global__ void paths_rk2(double *k,double *angles,double *pos);
__global__ void paths_rk4(double *k,double *angles,double *pos);

__device__ double f(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vy,double const &vz);
__device__ double g(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vz);
__device__ double gL(double const &t,double const &y,double const &z,double const &vz);
__device__ double h(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vy);
__device__ double hL(double const &t,double const &y,double const &z,double const &vy);

int main(){
	onHost();
	return 0;
}

void onHost(){
	FILE *k_vec,*posit=NULL;

	time_t rawtime;
	struct tm*timeinfo;

	time(&rawtime);
	timeinfo=localtime(&rawtime);

	printf("The current time is %s",asctime(timeinfo));
	
	const char* name_k="k-vectors";
	const char* name_p="screen";
	const char* format=".txt";

	char day[10];

	strftime(day, sizeof(day)-1, "%d_%H_%M", timeinfo);

	char strtmp[6];

	char filename_k[512];
	char filename_p[512];

	std::copy(asctime(timeinfo)+4,asctime(timeinfo)+7,strtmp);

	sprintf(filename_k,"%s%s%s%s",name_k,strtmp,day,format);
	sprintf(filename_p,"%s%s%s%s",name_p,strtmp,day,format);

	double *k_h,*theta_h,*phi_h; // Spherical coordinates for each k-mode (Nk in total)
	double *eta_h; // Random phases for the ZPF k-modes (Nk in total) <--- Following Boyer's work
	double *angles_h; // Single vector for the theta, phi and eta random numbers (3Nk in length for optimization purposes)
	double *xi_h; // Polarization angles in host space
	double *init_h; // Initial positions (h indicates host allocation)
	double *v_init_h; // Initial transverse velocities
	double *screen_h; // Single vector for the initial position, initial transverse velocities and final positions (3N in length for optimization purposes)

	k_h=(double*)malloc(Nk*sizeof(double));
	theta_h=(double*)malloc(Nk*sizeof(double));
	phi_h=(double*)malloc(Nk*sizeof(double));

	eta_h=(double*)malloc(Nk*sizeof(double));

	angles_h=(double*)malloc(3*Nk*sizeof(double));

	xi_h=(double*)malloc(Ne*sizeof(double));

	init_h=(double*)malloc(N*sizeof(double));

	v_init_h=(double*)malloc(N*sizeof(double));

	screen_h=(double*)malloc(3*N*sizeof(double));

	onDevice(k_h,theta_h,phi_h,eta_h,angles_h,xi_h,init_h,v_init_h,screen_h);

	k_vec=fopen(filename_k,"w");
	for(int i=0;i<Nk;i++){
		fprintf(k_vec,"%2.8e,%f,%f,%f\n",k_h[i],angles_h[i],angles_h[Nk+i],angles_h[2*Nk+i]);
	}
	fclose(k_vec);

	posit=fopen(filename_p,"w");
	for(int i=0;i<N;i++){
		fprintf(posit,"%2.6e,%2.6e,%2.6e\n",screen_h[i],screen_h[N+i],screen_h[2*N+i]);
	}
	fclose(posit);

	free(k_h);
	free(theta_h);
	free(phi_h);
	free(eta_h);
	free(angles_h);
	free(xi_h);
	free(init_h);
	free(v_init_h);
	free(screen_h);
}

void onDevice(double *k_h,double *theta_h,double *phi_h,double *eta_h,double *angles_h,double *xi_h,double *init_h,double *v_init_h,double *screen_h){
	unsigned int blocks=(Nk+TPB-1)/TPB;

	double pi_h=3.1415926535;
	double q_h=1.6e-19;
	double m_h=9.10938356e-31;
	double hbar_h=1.0545718e-34;
//	double hbar_h=0; // uncomment this line to see classical results
	double c_h=299792458.0;
	double eps0_h=8.85e-12;
	double v0_h=1.1e7;
	double fwhm_h=25e-6;
	double sigma_h=fwhm_h/(2.0*sqrt(2.0*log(2.0)));

	double lamL_h=532e-9;
	double kL_h=2*pi_h/lamL_h;
	double wL_h=kL_h*c_h;

	double sigma_p_h=4.0*pi_h*(1.0545718e-34)/(lamL_h*sqrt(2.0*log(2.0)));

	double lamR_h=lamL_h;
//	double lamR_h=2*pi_h*hbar_h/(m_h*v0_h);
	double kR_h=2*pi_h/lamR_h;
	double wR_h=kR_h*c_h;

	double E0L_h=1.7777777e8;
	double D_h=125e-6;
	double zimp_h=24e-2+D_h;
	double sigmaL_h=26e-6;

	double damping_h=6.245835e-24;
	double Delta_h=9e7*damping_h*pow(wR_h,2.0);
	double kmin_h=(wR_h-Delta_h/2.0)/c_h;
	double kmax_h=(wR_h+Delta_h/2.0)/c_h;
	double Vk_h=4.0*pi_h*(pow(kmax_h,3.0)-pow(kmin_h,3.0))/3.0;
	double V_h=pow(2.0*pi_h,3.0)*Nk/Vk_h;

	double dt_h=pi_h/(1.5*wR_h);
//	double dt_h=1.0/(0.1*wR_h);

	cudaMemcpyToSymbol(pi,&pi_h,sizeof(double));
	cudaMemcpyToSymbol(q,&q_h,sizeof(double));
	cudaMemcpyToSymbol(m,&m_h,sizeof(double));
	cudaMemcpyToSymbol(hbar,&hbar_h,sizeof(double));
	cudaMemcpyToSymbol(c,&c_h,sizeof(double));
	cudaMemcpyToSymbol(eps0,&eps0_h,sizeof(double));
	cudaMemcpyToSymbol(v0,&v0_h,sizeof(double));
	cudaMemcpyToSymbol(sigma,&sigma_h,sizeof(double));

	cudaMemcpyToSymbol(lamL,&lamL_h,sizeof(double));
	cudaMemcpyToSymbol(kL,&kL_h,sizeof(double));
	cudaMemcpyToSymbol(wL,&wL_h,sizeof(double));

	cudaMemcpyToSymbol(sigma_p,&sigma_p_h,sizeof(double));

	cudaMemcpyToSymbol(lamR,&lamR_h,sizeof(double));
	cudaMemcpyToSymbol(kR,&kR_h,sizeof(double));
	cudaMemcpyToSymbol(wR,&wR_h,sizeof(double));

	cudaMemcpyToSymbol(E0L,&E0L_h,sizeof(double));
	cudaMemcpyToSymbol(D,&D_h,sizeof(double));
	cudaMemcpyToSymbol(zimp,&zimp_h,sizeof(double));
	cudaMemcpyToSymbol(sigmaL,&sigmaL_h,sizeof(double));

	cudaMemcpyToSymbol(damping,&damping_h,sizeof(double));
	cudaMemcpyToSymbol(Delta,&Delta_h,sizeof(double));
	cudaMemcpyToSymbol(kmin,&kmin_h,sizeof(double));
	cudaMemcpyToSymbol(kmax,&kmax_h,sizeof(double));
	cudaMemcpyToSymbol(V,&V_h,sizeof(double));

	cudaMemcpyToSymbol(dt,&dt_h,sizeof(double));

	/* Polarization modes allocation (in CONSTANT memory) */
	for(int i=0;i<Ne;i++){
		xi_h[i]=i*2*pi_h/Ne;
	}

	cudaMemcpyToSymbol(xi,xi_h,Ne*sizeof(double));

	float elapsedTime; // Variables to record execution times
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double *k_d,*theta_d,*phi_d;
	double *eta_d;
	double *angles_d;
	double *init_d; // Vectors in Device (d indicates device allocation)
	double *v_init_d;
	double *screen_d;

	printf("Number of particles (N): %d\n",N);
	if(hbar_h>0.0){
		printf("Quantum version of the KD effect\n");
		printf("wR=%2.6e rad/s\n",wR_h);
		printf("Delta=%2.6e\n",Delta_h);
		printf("kmin=%2.6e\n",kmin_h);
		printf("kmax=%2.6e\n",kmax_h);
		printf("Vk=%2.6e\n",Vk_h);
		printf("V=%2.6e\n",V_h);
	}else printf("Classical version of the KD effect\n");
	printf("E0L=%2.6e V/m\n",E0L_h);
	printf("dt=%2.6e s\n",dt_h);
	printf("Number of k-modes (Nk): %d\n",Nk);
	printf("Number of polarizations (Ne): %d\n",Ne);
	printf("Threads per block: %d\n",TPB);
	printf("Number of blocks (k-modes): %d\n",blocks);

	cudaMalloc((void**)&k_d,Nk*sizeof(double));
	cudaMalloc((void**)&theta_d,Nk*sizeof(double));
	cudaMalloc((void**)&phi_d,Nk*sizeof(double));

	cudaMalloc((void**)&eta_d,Nk*sizeof(double));

	cudaMalloc((void**)&angles_d,3*Nk*sizeof(double));

	cudaMalloc((void**)&init_d,N*sizeof(double));

	cudaMalloc((void**)&v_init_d,N*sizeof(double));

	cudaMalloc((void**)&screen_d,3*N*sizeof(double));

	/* Randomly generated k-modes inside the spherical shell */

	cudaEventRecord(start,0);

	curandState *devStates_kmodes;
        cudaMalloc(&devStates_kmodes,Nk*sizeof(curandState));

	//k
	srand(time(0));
	int seed=rand(); //Setting up the seeds
	setup_kmodes<<<blocks,TPB>>>(devStates_kmodes,seed);

	kmodes<<<blocks,TPB>>>(k_d,devStates_kmodes,1,Nk);

	//theta
	kmodes<<<blocks,TPB>>>(theta_d,devStates_kmodes,2,Nk);

	//phi
	kmodes<<<blocks,TPB>>>(phi_d,devStates_kmodes,3,Nk);

	cudaMemcpy(k_h,k_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(theta_h,theta_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(phi_h,phi_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Random k-modes succesfully generated in %6.4f ms\n",elapsedTime);

	/* Randomly generated phases for the CPC modes */

	cudaEventRecord(start,0);

	curandState *devStates_eta;
	cudaMalloc(&devStates_eta,Nk*sizeof(curandState));

	blocks=(Nk+TPB-1)/TPB;
	printf("Number of blocks (phases): %d\n",blocks);

	//eta
	srand(time(NULL));
	seed=rand(); //Settin up seeds
	setup_kmodes<<<blocks,TPB>>>(devStates_eta,seed);

	kmodes<<<blocks,TPB>>>(eta_d,devStates_eta,6,Nk);

	cudaMemcpy(eta_h,eta_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Random ZPF phases succesfully generated in %6.4f ms\n",elapsedTime);

	/* Making a single vector for theta, phi and eta (reduces the size of memory, one double pointer instead of three) */
	
	for(int i=0;i<Nk;i++){
		angles_h[i]=theta_h[i];
		angles_h[Nk+i]=phi_h[i];
		angles_h[2*Nk+i]=eta_h[i];
		//angles_h[3*Nk+i]=eta_h[i+Nk];
	}

	cudaFree(theta_d);
	cudaFree(phi_d);
	cudaFree(eta_d);

	cudaMemcpy(angles_d,angles_h,3*Nk*sizeof(double),cudaMemcpyHostToDevice);

	/* Initial positions and transverse momentum*/

	cudaEventRecord(start,0);

	curandState *devStates_init;
	cudaMalloc(&devStates_init,N*sizeof(curandState));

	blocks=(N+TPB-1)/TPB;
	printf("Number of blocks (paths): %d\n",blocks);

	srand(time(NULL));
	seed=rand();
	setup_kmodes<<<blocks,TPB>>>(devStates_init,seed);

	kmodes<<<blocks,TPB>>>(init_d,devStates_init,4,N);
	cudaMemcpy(init_h,init_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	kmodes<<<blocks,TPB>>>(v_init_d,devStates_init,5,N);
	cudaMemcpy(v_init_h,v_init_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	/* Making a single vector for the initial and final positions (reduces the size of memory, one double pointer instead of two) */

	for(int i=0;i<N;i++){
		screen_h[i]=init_h[i];
		screen_h[N+i]=v_init_h[i];
		screen_h[2*N+i]=0.0;
	}

	cudaMemcpy(screen_d,screen_h,3*N*sizeof(double),cudaMemcpyHostToDevice);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Positions vector initialized in %6.4f ms\n",elapsedTime);

	cudaEventRecord(start,0);

	//paths_euler<<<blocks,TPB>>>(k_d,angles_d,screen_d);
	//paths_rk2<<<blocks,TPB>>>(k_d,angles_d,screen_d);
	paths_rk4<<<blocks,TPB>>>(k_d,angles_d,screen_d);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	//printf("Paths computed using Euler method in %6.4f hours\n",elapsedTime*1e-3/3600.0);
	//printf("Paths computed using RK2 method in %6.4f hours\n",elapsedTime*1e-3/3600.0);
	printf("Paths computed using RK4 method in %6.4f hours\n",elapsedTime*1e-3/3600.0);
	printf("------------------------------------------------------------\n");

	cudaMemcpy(screen_h,screen_d,3*N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(devStates_kmodes);
	cudaFree(devStates_eta);
	cudaFree(devStates_init);
	cudaFree(k_d);
	cudaFree(angles_d);
	cudaFree(init_d);
	cudaFree(v_init_d);
	cudaFree(screen_d);
}

__global__ void setup_kmodes(curandState *state,unsigned long seed){
        int idx=threadIdx.x+blockIdx.x*blockDim.x;
        curand_init(seed,idx,0,&state[idx]);
}

__global__ void kmodes(double *vec,curandState *globalState,int opt,int n){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	curandState localState=globalState[idx];
	if(idx<n){
		if(opt==1){ // Random radii
			if(idx==0){
				vec[idx]=kL/2.0;
			}else if(idx==1){
				vec[idx]=kL;
			}else if(idx==2){
				vec[idx]=3.0*kL/2.0;
			}else{
				vec[idx]=pow((pow(kmax,3.0)-pow(kmin,3.0))*curand_uniform(&localState)+pow(kmin,3.0),1.0/3.0);
			}
			//vec[idx]=pow((pow(kmax,3.0)-pow(kmin,3.0))*curand_uniform(&localState)+pow(kmin,3.0),1.0/3.0);
		}else if(opt==2){ // Random polar angles
			if(idx==0||idx==1||idx==2){
				vec[idx]=pi/2.0;
			}else{
				vec[idx]=acos(1.0-2.0*curand_uniform(&localState));
			}
			//vec[idx]=acos(1.0-2.0*curand_uniform(&localState));
		}else if(opt==3){ // Random azimuthal angles
			if(idx==0||idx==1||idx==2){
				vec[idx]=pi/2.0;
			}else{
				vec[idx]=2.0*pi*curand_uniform(&localState);
			}
			//vec[idx]=2.0*pi*curand_uniform(&localState);
		}else if(opt==4){
			vec[idx]=sigma*curand_normal(&localState); // Random initial positions
		}else if(opt==5){
			vec[idx]=sigma_p*curand_normal(&localState); // Random initial transverse momentum
		}else if(opt==6){
			if(idx==0||idx==1||idx==2){
				vec[idx]=0.001;
			}else{
				vec[idx]=2.0*pi*curand_uniform(&localState); // Random constant phases
			}
			//vec[idx]=2.0*pi*curand_uniform(&localState);
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

		vxnn[threadIdx.x]=0.0;
		vynn[threadIdx.x]=0.0;
		vznn[threadIdx.x]=0.0;

		while(zn<=D){
			for(int i=0;i<Nk;i++){
				for(int j=0;j<Ne;j++){
					__syncthreads();
					vxnn[threadIdx.x]=vxnn[threadIdx.x]+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vyn,vzn); // vxnn represents here the total ZPF force in x (recycled variable)
					__syncthreads();
					vynn[threadIdx.x]=vynn[threadIdx.x]+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vxn,vzn); // k1vy represents here the total ZPF force in y
					__syncthreads();
					vznn[threadIdx.x]=vznn[threadIdx.x]+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vxn,vyn); // k1vz represents here the total ZPF force in z
				}
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
		}
		__syncthreads();
		pos[2*N+idx]=yn+(zimp-D)*vyn/vzn;
	}
}
__global__ void paths_rk2(double *k,double *angles,double *pos){
	unsigned int idx=threadIdx.x+blockIdx.x*TPB;

	__shared__ double k1vx[TPB];
	__shared__ double k1vy[TPB];
	__shared__ double k1vz[TPB];
	__shared__ double k2vx[TPB];
	__shared__ double k2vy[TPB];
	__shared__ double k2vz[TPB];

	if(idx<N){
		double tn=0.0;
		double xn=0.0;
		double yn=pos[idx];
		double zn=0.0;

		double vxn=0.0;
		double vyn=pos[N+idx];
		__syncthreads();
		double vzn=v0;

		double vxnn=0.0;
		double vynn=0.0;
		double vznn=0.0;

		while(zn<=D){ // Only laser region. After the particle leaves it, the final position is extrapolated
			k1vx[threadIdx.x]=0.0;
			k1vy[threadIdx.x]=0.0;
			k1vz[threadIdx.x]=0.0;
			k2vx[threadIdx.x]=0.0;
			k2vy[threadIdx.x]=0.0;
			k2vz[threadIdx.x]=0.0;

			for(int i=0;i<Nk;i++){
				for(int j=0;j<Ne;j++){
					__syncthreads();
					k1vx[threadIdx.x]=k1vx[threadIdx.x]+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vyn,vzn); // k1vx represents here the total ZPF force in x
					__syncthreads();
					k1vy[threadIdx.x]=k1vy[threadIdx.x]+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vxn,vzn); // k1vy represents here the total ZPF force in y
					__syncthreads();
					k1vz[threadIdx.x]=k1vz[threadIdx.x]+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vxn,vyn); // k1vz represents here the total ZPF force in z
				}
			}

			k1vy[threadIdx.x]=k1vy[threadIdx.x]+gL(tn,yn,zn,vzn); // Laser contribution to the total force in y
			k1vz[threadIdx.x]=k1vz[threadIdx.x]+hL(tn,yn,zn,vyn); // Laser contribution to the total force in z

			for(int i=0;i<Nk;i++){
				for(int j=0;j<Ne;j++){
					__syncthreads();
					k2vx[threadIdx.x]=k2vx[threadIdx.x]+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn+dt,xn+dt*vxn,yn+dt*vyn,zn+dt*vzn,vyn+dt*k1vy[threadIdx.x],vzn+dt*k1vz[threadIdx.x]); // k2vx represents here the total ZPF force in x
					__syncthreads();
					k2vy[threadIdx.x]=k2vy[threadIdx.x]+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn+dt,xn+dt*vxn,yn+dt*vyn,zn+dt*vzn,vxn+dt*k1vx[threadIdx.x],vzn+dt*k1vz[threadIdx.x]); // k2vy represents here the total ZPF force in y
					__syncthreads();
					k2vz[threadIdx.x]=k2vz[threadIdx.x]+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn+dt,xn+dt*vxn,yn+dt*vyn,zn+dt*vzn,vxn+dt*k1vx[threadIdx.x],vyn+dt*k1vy[threadIdx.x]); // k2vz represents here the total ZPF force in z
				}
			}

			__syncthreads();
			k2vy[threadIdx.x]=k2vy[threadIdx.x]+gL(tn+dt,yn+dt*vyn,zn+dt*vzn,vzn+dt*k1vz[threadIdx.x]); // Laser contribution to the total force in y
			__syncthreads();
			k2vz[threadIdx.x]=k2vz[threadIdx.x]+hL(tn+dt,yn+dt*vyn,zn+dt*vzn,vyn+dt*k1vy[threadIdx.x]); // Laser contribution to the total force in z

			__syncthreads();
			vxnn=vxn+dt*(k1vx[threadIdx.x]+k2vx[threadIdx.x])/2.0;
			__syncthreads();
			vynn=vyn+dt*(k1vy[threadIdx.x]+k2vy[threadIdx.x])/2.0;
			__syncthreads();
			vznn=vzn+dt*(k1vz[threadIdx.x]+k2vz[threadIdx.x])/2.0;

			__syncthreads();
			xn=xn+dt*(vxn+vxnn)/2.0;
			__syncthreads();
			yn=yn+dt*(vyn+vynn)/2.0;
			__syncthreads();
			zn=zn+dt*(vzn+vznn)/2.0;
		}
		__syncthreads();
		pos[2*N+idx]=yn+(zimp-D)*vyn/vzn;
	}
}
__global__ void paths_rk4(double *k,double *angles,double *pos){
	unsigned int idx=threadIdx.x+blockIdx.x*TPB;

	__shared__ double k1vx[TPB];
	__shared__ double k1vy[TPB];
	__shared__ double k1vz[TPB];
	
	__shared__ double k2vx[TPB];
	__shared__ double k2vy[TPB];
	__shared__ double k2vz[TPB];
	
	__shared__ double k3vx[TPB];
	__shared__ double k3vy[TPB];
	__shared__ double k3vz[TPB];

	if(idx<N){
		double tn=0.0;

		double xn=0.0;
		double yn=pos[idx];
		double zn=0.0;

		double vxn=0.0;
		double vyn=pos[N+idx];;
		__syncthreads();
		double vzn=v0;

		double k1x;
		double k1y;
		double k1z;

		double k2x;
		double k2y;
		double k2z;

		double k3x;
		double k3y;
		double k3z;

		double k4x;
		double k4y;
		double k4z;

		double k4vx;
		double k4vy;
		double k4vz;

		while(zn<=D){
			k1x=0.0;
			k1y=0.0;
			k1z=0.0;

			k2x=0.0;
			k2y=0.0;
			k2z=0.0;
			
			k3x=0.0;
			k3y=0.0;
			k3z=0.0;

			k4x=0.0;
			k4y=0.0;
			k4z=0.0;

			k1vx[threadIdx.x]=0.0;
			k1vy[threadIdx.x]=0.0;
			k1vz[threadIdx.x]=0.0;

			k2vx[threadIdx.x]=0.0;
			k2vy[threadIdx.x]=0.0;
			k2vz[threadIdx.x]=0.0;

			k3vx[threadIdx.x]=0.0;
			k3vy[threadIdx.x]=0.0;
			k3vz[threadIdx.x]=0.0;

			k4vx=0.0;
			k4vy=0.0;
			k4vz=0.0;

			for(int i=0;i<Nk;i++){ // k1
				for(int j=0;i<Ne;i++){
					__syncthreads();
					k1vx[threadIdx.x]=k1vx[threadIdx.x]+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vyn,vzn); // k1vx represents here the total ZPF force in x
					__syncthreads();
					k1vy[threadIdx.x]=k1vy[threadIdx.x]+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vxn,vzn); // k1vy represents here the total ZPF force in y
					__syncthreads();
					k1vz[threadIdx.x]=k1vz[threadIdx.x]+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn,yn,zn,vxn,vyn); // k1vz represents here the total ZPF force in z
				}
			}
			__syncthreads();
			k1vy[threadIdx.x]=k1vy[threadIdx.x]+gL(tn,yn,zn,vzn); // Laser contribution to the total force in y
			__syncthreads();
			k1vz[threadIdx.x]=k1vz[threadIdx.x]+hL(tn,yn,zn,vyn); // Laser contribution to the total force in z

			__syncthreads();
			tn=tn+dt/2.0;

			k1x=vxn;
			k1y=vyn;
			k1z=vzn;
			
			for(int i=0;i<Nk;i++){ // k2
				for(int j=0;j<Ne;j++){
					__syncthreads();
					k2vx[threadIdx.x]=k2vx[threadIdx.x]+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k1x/2.0,yn+dt*k1y/2.0,zn+dt*k1z/2.0,vyn+dt*k1vy[threadIdx.x]/2.0,vzn+dt*k1vz[threadIdx.x]/2.0); // k2vx represents here the total force in x
					__syncthreads();
					k2vy[threadIdx.x]=k2vy[threadIdx.x]+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k1x/2.0,yn+dt*k1y/2.0,zn+dt*k1z/2.0,vxn+dt*k1vx[threadIdx.x]/2.0,vzn+dt*k1vz[threadIdx.x]/2.0); // k2vy represents here the total force in y
					__syncthreads();
					k2vz[threadIdx.x]=k2vz[threadIdx.x]+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k1x/2.0,yn+dt*k1y/2.0,zn+dt*k1z/2.0,vxn+dt*k1vx[threadIdx.x]/2.0,vyn+dt*k1vy[threadIdx.x]/2.0); // k2vz represents here the total force in z
				}
			}
			__syncthreads();
			k2vy[threadIdx.x]=k2vy[threadIdx.x]+gL(tn,yn+dt*k1y/2.0,zn+dt*k1z/2.0,vzn+dt*k1vz[threadIdx.x]/2.0); // Laser contribution to the total force in y
			__syncthreads();
			k2vz[threadIdx.x]=k2vz[threadIdx.x]+hL(tn,yn+dt*k1y/2.0,zn+dt*k1z/2.0,vyn+dt*k1vy[threadIdx.x]/2.0); // Laser contribution to the total force in z

			__syncthreads();
			k2x=vxn+dt*k1vx[threadIdx.x]/2.0;
			__syncthreads();
			k2y=vyn+dt*k1vy[threadIdx.x]/2.0;
			__syncthreads();
			k2z=vzn+dt*k1vz[threadIdx.x]/2.0;

			for(int i=0;i<Nk;i++){ // k3
				for(int j=0;j<Ne;j++){
					__syncthreads();
					k3vx[threadIdx.x]=k3vx[threadIdx.x]+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k2x/2.0,yn+dt*k2y/2.0,zn+dt*k2z/2.0,vyn+dt*k2vy[threadIdx.x]/2.0,vzn+dt*k2vz[threadIdx.x]/2.0); // k3vx represents here the total force in x
					__syncthreads();
					k3vy[threadIdx.x]=k3vy[threadIdx.x]+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k2x/2.0,yn+dt*k2y/2.0,zn+dt*k2z/2.0,vxn+dt*k2vx[threadIdx.x]/2.0,vzn+dt*k2vz[threadIdx.x]/2.0); // k3vy represents here the total force in y
					__syncthreads();
					k3vz[threadIdx.x]=k3vz[threadIdx.x]+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k2x/2.0,yn+dt*k2y/2.0,zn+dt*k2z/2.0,vxn+dt*k2vx[threadIdx.x]/2.0,vyn+dt*k2vy[threadIdx.x]/2.0); // k3vz represents here the total force in z
				}
			}
			__syncthreads();
			k3vy[threadIdx.x]=k3vy[threadIdx.x]+gL(tn,yn+dt*k2y/2.0,zn+dt*k2z/2.0,vzn+dt*k2vz[threadIdx.x]/2.0); // Laser contribution to the total force in y
			__syncthreads();
			k3vz[threadIdx.x]=k3vz[threadIdx.x]+hL(tn,yn+dt*k2y/2.0,zn+dt*k2z/2.0,vyn+dt*k2vy[threadIdx.x]/2.0); // Laser contribution to the total force in z

			__syncthreads();
			tn=tn+dt/2.0;

			__syncthreads();
			k3x=vxn+dt*k2vx[threadIdx.x]/2.0;
			__syncthreads();
			k3y=vyn+dt*k2vy[threadIdx.x]/2.0;
			__syncthreads();
			k3z=vzn+dt*k2vz[threadIdx.x]/2.0;

			for(int i=0;i<Nk;i++){ // k4
				for(int j=0;j<Ne;j++){
					__syncthreads();
					k4vx=k4vx+f(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k3x,yn+dt*k3y,zn+dt*k3z,vyn+dt*k3vy[threadIdx.x],vzn+dt*k3vz[threadIdx.x]); // k4vx represents here the total force in x
					__syncthreads();
					k4vy=k4vy+g(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k3x,yn+dt*k3y,zn+dt*k3z,vxn+dt*k3vx[threadIdx.x],vzn+dt*k3vz[threadIdx.x]); // k4vy represents here the total force in y
					__syncthreads();
					k4vz=k4vz+h(k[i],angles[i],angles[Nk+i],angles[2*Nk+i],xi[j],tn,xn+dt*k3x,yn+dt*k3y,zn+dt*k3z,vxn+dt*k3vx[threadIdx.x],vyn+dt*k3vy[threadIdx.x]); // k4vz represents here the total force in z
				}
			}
			k4vy=k4vy+gL(tn,yn+dt*k3y,zn+dt*k3z,vzn+dt*k3vz[threadIdx.x]); // Laser contribution to the total force in y
			k4vz=k4vz+hL(tn,yn+dt*k3y,zn+dt*k3z,vyn+dt*k3vy[threadIdx.x]); // Laser contribution to the total force in z

			__syncthreads();
			xn=xn+dt*(k1x+2.0*k2x+2.0*k3x+k4x)/6.0;
			__syncthreads();
			yn=yn+dt*(k1y+2.0*k2y+2.0*k3y+k4y)/6.0;
			__syncthreads();
			zn=zn+dt*(k1z+2.0*k2z+2.0*k3z+k4z)/6.0;

			__syncthreads();
			vxn=vxn+dt*(k1vx[threadIdx.x]+2.0*k2vx[threadIdx.x]+2.0*k3vx[threadIdx.x]+k4vx)/6.0;
			__syncthreads();
			vyn=vyn+dt*(k1vy[threadIdx.x]+2.0*k2vy[threadIdx.x]+2.0*k3vy[threadIdx.x]+k4vy)/6.0;
			__syncthreads();
			vzn=vzn+dt*(k1vz[threadIdx.x]+2.0*k2vz[threadIdx.x]+2.0*k3vz[threadIdx.x]+k4vz)/6.0;
		}
		__syncthreads();
		pos[2*N+idx]=yn+(zimp-D)*vyn/vzn;
	}
}
/*
__device__ double f(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vy,double const &vz){ // ZPF, x-component (Wayne-Herman version)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;

	__syncthreads();
	double E0=10.0*sqrt(hbar*w/(eps0*V));

	__syncthreads();
	return q*E0*(cos(phi1)+cos(phi2))*(cos(theta)*cos(phi)*cos(xi)-sin(phi)*sin(xi))/m+q*E0*(cos(phi1)-cos(phi2))*(sin(theta)*sin(xi)*vy+(cos(theta)*sin(phi)*sin(xi)-cos(phi)*cos(xi))*vz)/(m*c);
}
*/
/*
__device__ double f(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vy,double const &vz){ // ZPF, x-component (WH+multiple Fourier modes)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;

	double E0;
	if(k==kR/2.0){
		__syncthreads();
		E0=sqrt(2.0/(3.0*pi))*E0L;
	}else if(k==kR){
		__syncthreads();
		E0=sqrt(1.0/2.0)*E0L;
	}else if(k==3.0*kR/2.0){
		__syncthreads();
		E0=sqrt(6.0/(5.0*pi))*E0L;
	}else{
		__syncthreads();
		E0=sqrt(hbar*w/(eps0*V));
	}
	return 0.0;
}
*/

__device__ double f(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vy,double const &vz){ // ZPF, x-component (Single explicit Fourier mode)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*y+eta;
	__syncthreads();
	double phi2=w*t+k*y+eta;

	double E0;
	if(k==kR/2.0){
		__syncthreads();
		E0=sqrt(2.0/(3.0*pi))*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0))); // SHOULD I USE LASER GAUSSIAN PROFILE HERE OR NOT?
	}else if(k==kR){
		__syncthreads();
		E0=sqrt(1.0/2.0)*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else if(k==3.0*kR/2.0){
		__syncthreads();
		E0=sqrt(6.0/(5.0*pi))*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else{
		__syncthreads();
		E0=sqrt(hbar*w/(eps0*V));
	}
	//__syncthreads();
	//return q*E0*(cos(phi1)-cos(phi2))*(sin(theta)*sin(xi)*vy+(cos(theta)*sin(phi)*sin(xi)-cos(phi)*cos(xi))*vz)/(m*c);
	return 0.0;
}

/*
__device__ double g(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vz){ // ZPF, y-component (Wayne-Herman version)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;

	__syncthreads();
	double E0=10.0*sqrt(hbar*w/(eps0*V));

	__syncthreads();
	return q*E0*(cos(phi1)+cos(phi2))*(cos(theta)*sin(phi)*cos(xi)+cos(phi)*sin(xi))/m-q*E0*(cos(phi1)-cos(phi2))*(sin(theta)*sin(xi)*vx+(cos(theta)*cos(phi)*sin(xi)+sin(phi)*cos(xi))*vz)/(m*c);
}
*/
/*
__device__ double g(double &kv,double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vz){ // ZPF, y-component (WH+multiple Fourier modes)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;

	double E0;
	if(k==kR/2.0){
		__syncthreads();
		E0=sqrt(2.0/(3.0*pi))*E0L;
	}else if(k==kR){
		__syncthreads();
		E0=(1.0/sqrt(2.0))*E0L;
	}else if(k==3.0*kR/2.0){
		__syncthreads();
		E0=sqrt(6.0/(5.0*pi))*E0L;
	}else{
		__syncthreads();
		E0=sqrt(hbar*w/(eps0*V));
	}
	__syncthreads();
	return -q*E0*(cos(phi1)-cos(phi2))*vz/(m*c);
}
*/

__device__ double g(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vz){ // ZPF, y-component (Single explicit Fourier mode)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*y+eta;
	__syncthreads();
	double phi2=w*t+k*y+eta;

	double E0;
	if(k==kR/2.0){
		__syncthreads();
		E0=sqrt(2.0/(3.0*pi))*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else if(k==kR){
		__syncthreads();
		E0=sqrt(1.0/2.0)*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else if(k==3.0*kR/2.0){
		__syncthreads();
		E0=sqrt(6.0/(5.0*pi))*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else{
		__syncthreads();
		E0=sqrt(hbar*w/(eps0*V));
	}
	__syncthreads();
	return -q*E0*(cos(phi1)-cos(phi2))*vz/(m*c);
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
/*
__device__ double h(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vy){ // ZPF, z-component (Wayne-Herman version)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	
	__syncthreads();
	double E0=10.0*sqrt(hbar*w/(eps0*V));

	__syncthreads();
	return -q*E0*(cos(phi1)+cos(phi2))*(sin(theta)*cos(xi))/m+q*E0*(cos(phi1)-cos(phi2))*((cos(phi)*cos(xi)-cos(theta)*sin(phi)*sin(xi))*vx+(sin(phi)*cos(xi)+cos(theta)*cos(phi)*sin(xi))*vy)/(m*c);
}
*/
/*
__device__ double h(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vy){ // ZPF, z-component (WH+multiple Fourier modes)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	
	double E0
	if(k==kR/2.0){
		__syncthreads();
		E0=sqrt(2.0/(3.0*pi))*E0L;
	}else if(k==kR){
		__syncthreads();
		E0=(1/sqrt(2.0))*E0L;
	}else if(k==3.0*kR/2.0){
		__syncthreads();
		E0=sqrt(6.0/(5.0*pi))*E0L;
	}else{
		__syncthreads();
		E0=sqrt(hbar*w/(eps0*V));
	}
	__syncthreads();
	return -q*E0*(cos(phi1)+cos(phi2))/m+q*E0*(cos(phi1)-cos(phi2))*vy/(m*c);
}
*/

__device__ double h(double const &k,double const &theta,double const &phi,double const &eta,double &xi,double const &t,double const &x,double const &y,double const &z,double const &vx,double const &vy){ // ZPF, z-component (Single explicit Fourier mode)
	__syncthreads();
	double w=k*c;

	__syncthreads();
	//double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	double phi1=w*t-k*y+eta;
	__syncthreads();
	//double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta;
	double phi2=w*t+k*y+eta;
	
	double E0;
	if(k==kR/2.0){
		__syncthreads();
		E0=sqrt(2.0/(3.0*pi))*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else if(k==kR){
		__syncthreads();
		E0=sqrt(1.0/2.0)*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else if(k==3.0*kR/2.0){
		__syncthreads();
		E0=sqrt(6.0/(5.0*pi))*E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	}else{
		__syncthreads();
		E0=sqrt(hbar*w/(eps0*V));
	}
	__syncthreads();
	return -q*E0*(cos(phi1)+cos(phi2))/m+q*E0*(cos(phi1)-cos(phi2))*vy/(m*c);
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
