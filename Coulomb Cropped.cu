
void onDevice(double *r_h,double *theta_h,double *phi_h,double *p_h,double *theta_p_h,double *phi_p_h,double *E_h,double *pos_h,double *mom_h){
	unsigned int blocks=(N+TPB-1)/TPB; // Check this line for optimization purposes

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
	
	cudaMemcpy(pos_h,r,3*N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(mom_h,p,3*N*sizeof(double),cudaMemcpyDeviceToHost);

	Efield<<<blocks,TPB>>>(r,E);
	
	//E field GPU to CPU migration(for debugging only)
	cudaMemcpy(E_h,E,3*N*sizeof(double),cudaMemcpyDeviceToHost);

	paths_euler<<<blocks,TPB>>>(r,p,E);

	int dsizes=10*steps*N;

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
		for(unsigned i=0;i<results.size()-1;i=i+10){
			if(results[i]+results[i+1]!=0){
				myfile << std::scientific << results[i] << ',' << results[i+1] << ',' << results[i+2]  << ',' << results[i+3]  << ',' << results[i+4]  << ',' << results[i+5] << ',' << results[i+6] << ',' << results[i+7] << ',' << results[i+8] << ',' << std::defaultfloat << static_cast<int>(results[i+9]) << '\n';
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
			vec[idx]=sigma_p*curand_uniform(&localState); // Arjun said that he doesn't see why |p| should have any preference between 0 and 1eV
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
		/*if(idx==0){
			vec[3*idx+1]=0.01*rmax;
		}else{
			vec[3*idx+1]=-0.01*rmax;
		}
		vec[3*idx]=0;
		__syncthreads();
		vec[3*idx+2]=rtip+rmax;*/
	}
}

__global__ void Efield(double *pos,double *E){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;

	//double R1,R2;

	E[3*idx]=0;
	E[3*idx+1]=0;
	E[3*idx+2]=0;
	if(idx<N){
		for(int i=0;i<N;i++){ // Comment/uncomment this for cycle to disable/enable the Coulomb repulsion between charges, as well as in line 483
			if(i!=idx){
				__syncthreads();
				E[3*idx]=E[3*idx]+k*q*(pos[3*idx]-pos[3*i])/pow(pow(pos[3*idx]-pos[3*i],2.0)+pow(pos[3*idx+1]-pos[3*i+1],2.0)+pow(pos[3*idx+2]-pos[3*i+2],2.0),3.0/2.0);
				__syncthreads();
				E[3*idx+1]=E[3*idx+1]+k*q*(pos[3*idx+1]-pos[3*i+1])/pow(pow(pos[3*idx]-pos[3*i],2.0)+pow(pos[3*idx+1]-pos[3*i+1],2.0)+pow(pos[3*idx+2]-pos[3*i+2],2.0),3.0/2.0);
				__syncthreads();
				E[3*idx+2]=E[3*idx+2]+k*q*(pos[3*idx+2]-pos[3*i+2])/pow(pow(pos[3*idx]-pos[3*i],2.0)+pow(pos[3*idx+1]-pos[3*i+1],2.0)+pow(pos[3*idx+2]-pos[3*i+2],2.0),3.0/2.0);
			}
		}
		// Radial Electric field from the tip
		__syncthreads();
		E[3*idx]=E[3*idx]+rtip*Vtip*pos[3*idx]/pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2],2.0),3.0/2.0);
		__syncthreads();
		E[3*idx+1]=E[3*idx+1]+rtip*Vtip*pos[3*idx+1]/pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2],2.0),3.0/2.0);
		__syncthreads();
		E[3*idx+2]=E[3*idx+2]+rtip*Vtip*pos[3*idx+2]/pow(pow(pos[3*idx],2.0)+pow(pos[3*idx+1],2.0)+pow(pos[3*idx+2],2.0),3.0/2.0);

		// Electric field from the tip using method of images
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

	if(idx<N){
		double tn=0.0;

		/*double vxn=p[3*idx]/m;
		double vyn=p[3*idx+1]/m;
		double vzn=p[3*idx+2]/m;*/
		double vxn=0;
		double vyn=0;
		double vzn=0;

		//double R1,R2;

		while(r[3*idx+2]<=zdet && iter<steps){
			my_push_back(r[3*idx],r[3*idx+1],r[3*idx+2],vxn,vyn,vzn,E[3*idx],E[3*idx+1],E[3*idx+2],idx);

			vxn=vxn+dt*q*E[3*idx]/m; // minus sign to account for the e charge
			vyn=vyn+dt*q*E[3*idx+1]/m;
			vzn=vzn+dt*q*E[3*idx+2]/m;

			tn=tn+dt;

			r[3*idx]=r[3*idx]+dt*vxn;
			r[3*idx+1]=r[3*idx+1]+dt*vyn;
			r[3*idx+2]=r[3*idx+2]+dt*vzn;

			for(int i=0;i<N;i++){ // Comment/uncomment this for cycle to disable/enable the Coulomb repulsion between charges, as well as in line 375
				if(i!=idx && r[3*i+2]<zdet){
					E[3*idx]=E[3*idx]+k*q*(r[3*idx]-r[3*i])/pow(pow(r[3*idx]-r[3*i],2.0)+pow(r[3*idx+1]-r[3*i+1],2.0)+pow(r[3*idx+2]-r[3*i+2],2.0),3.0/2.0);
					E[3*idx+1]=E[3*idx+1]+k*q*(r[3*idx+1]-r[3*i+1])/pow(pow(r[3*idx]-r[3*i],2.0)+pow(r[3*idx+1]-r[3*i+1],2.0)+pow(r[3*idx+2]-r[3*i+2],2.0),3.0/2.0);
					E[3*idx+2]=E[3*idx+2]+k*q*(r[3*idx+2]-r[3*i+2])/pow(pow(r[3*idx]-r[3*i],2.0)+pow(r[3*idx+1]-r[3*i+1],2.0)+pow(r[3*idx+2]-r[3*i+2],2.0),3.0/2.0);
				}
			}
			// Radial Electric field from the tip
			E[3*idx]=E[3*idx]+rtip*Vtip*r[3*idx]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0);
			E[3*idx+1]=E[3*idx+1]+rtip*Vtip*r[3*idx+1]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0);
			E[3*idx+2]=E[3*idx+2]+rtip*Vtip*r[3*idx+2]/pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),3.0/2.0);

			// Electric field from the tip using method of images
			/*R1=pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2],2.0),1.0/2.0);
			__syncthreads();
			R2=pow(pow(r[3*idx],2.0)+pow(r[3*idx+1],2.0)+pow(r[3*idx+2]-2.0*zdet,2.0),1.0/2.0);
			__syncthreads();
			E[3*idx]=E[3*idx]+Vtip*r[3*idx]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));
			__syncthreads();
			E[3*idx+1]=E[3*idx+1]+Vtip*r[3*idx+1]*(1.0/pow(R1,3.0)-1.0/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));
			__syncthreads();
			E[3*idx+2]=E[3*idx+2]+Vtip*((r[3*idx+2]-2.0*zdet)/pow(R1,3.0)-r[3*idx+2]/pow(R2,3.0))/(1.0/rtip-1.0/(2.0*zdet));*/

			++iter;
			__syncthreads();
		}
	}
}
