#include "AssomOnline.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "ToMatlab.h"
#include <math.h>
#include<float.h>
#include <iostream>

__global__ void globalNormalizeBases(float* bases);
__global__ void globalNormalizeProb(float* nodeProb);
__global__ void globalExtactElements(float*src,float* dst, int*index);
__global__ void globalGetWinProj(float*src,float*coef,float*error, int*index);
__global__ void globalUpdateStep1(float* w_coeff1,float* w_coeff2,int*winner,float*coef1,float*coef2 , float*proj, float sigma_h);
__global__ void globalCalcUpdate(float *bases1,float *bases2, float* interSum, float* winput1, float* winput2,float alpha );
__global__ void globalFindMax(int*maxIndex, float*data);
__global__ void globalBlockMulAdd(float*dst,float* mat1, float*mat2);
__global__ void globalGetEmission(float*emsission, float*proj,float* normSigmaW, float* normSigmaN);
__global__ void globalGetWinErr(float*src,float*error, int*index);
__global__ void globalGetWinErrProj(float*src,float*error,float *winProj, int*index);

__device__ void deviceNormalizeBases(float* bases,volatile float * sbases,const unsigned int& x, const unsigned int& y);
__device__ void deviceDotProject(const float* sbases1,volatile float* sbases2,const unsigned int& x);
__device__ void deviceNormalizeProb(float* nodeProb, const float* sprob,const unsigned int& x,const unsigned int& y);
__device__ void deviceMulSum(float* dst, volatile float* svec1,  volatile float* svec2,const unsigned int& x, const unsigned int& y, const unsigned int& z  );

__device__ float cumax(float a, float b)
{
	return a > b ? a : b;
}
__global__ void globalNormalizeBases(float* bases1,float* bases2){

	const unsigned int y = blockIdx.y;
	const unsigned int x = threadIdx.x;
	const unsigned int ind = y*BASISDIM + x;
	__shared__ float sbases1[BASISDIM];
	__shared__ float sbases2[BASISDIM];
	sbases1[x] = bases1[ind];
	sbases2[x] = bases2[ind];
	if(x<BASISDIM-BLOCKSIZE){
		sbases1[x+BLOCKSIZE] = bases1[ind+BLOCKSIZE];
		sbases2[x+BLOCKSIZE] = bases2[ind+BLOCKSIZE];
	}
	__syncthreads();
	deviceNormalizeBases(bases1,sbases1,x,y);
	deviceDotProject(sbases1,sbases2,x);
	deviceNormalizeBases(bases2,sbases2,x,y);
}


//give the 
__global__ void globalGetWinErr(float*src,float*error, int*index)
{
	const unsigned int x = threadIdx.x;
	float projVal;

	if(x<AGENTSNUM)
	{
		projVal = src[index[x]+x*BASESNUM];
		error[x] =INPUTSQNORM-projVal;	

	}
	__syncthreads();


}

//return the winner projection and error
__global__ void globalGetWinErrProj(float*src,float*error,float *winProj, int*index)
{
	const unsigned int x = threadIdx.x;
	float projVal;

	if(x<AGENTSNUM)
	{
		projVal = src[index[x]+x*BASESNUM];
		error[x] =INPUTSQNORM-projVal;	
		winProj[x] = projVal; 
	}
	__syncthreads();


}


__global__ void globalGetWinProj(float*src,float*coef,float*error, int*index)
{
	const unsigned int x = threadIdx.x;
	__shared__ float sdata[BASESNUM];
	float projVal;

	sdata[x]=0;
	if(x<BASESNUM-BLOCKSIZE2)
		sdata[x+BLOCKSIZE2]=0;

	if(x<AGENTSNUM)
	{
		projVal = src[index[x]+x*BASESNUM];
		error[x] =INPUTSQNORM-projVal;
		atomicAdd(&sdata[index[x]],projVal);
		

	}
	__syncthreads();

	coef[x] = sdata[x]/AGENTSNUM;
	if(x<BASESNUM-BLOCKSIZE2)
		coef[x+BLOCKSIZE2] = sdata[x+BLOCKSIZE2]/AGENTSNUM;
}

__global__ void globalGetEmission(float*emission, float*proj,float normSigmaW, float normSigmaN)
{
	const unsigned int y = blockIdx.y; //each block gets all projs of an observation, i.e. 100 blocks and 400 threads
	const unsigned int x = threadIdx.x;
	const unsigned int ind = y*BASESNUM + x;

	float pval, eval, emiss; 

	if (x<BASESNUM)
	{
		pval = proj[ind];
		eval = INPUTSQNORM - pval;
		emiss = exp(-(normSigmaW*pval+normSigmaN*eval));
		emission[ind] = emiss;
	}


}


__global__ void globalExtactElements(float*src,float* dst, int*index)
{
	const unsigned int y = blockIdx.y;
	const unsigned int x = threadIdx.x;
	const unsigned int ind = y*BASESNUM + x;

	dst[ind] = src[index[ind]+x*BASESNUM+y*BASESNUM*BASESNUM];
	if(x<BASESNUM-BLOCKSIZE2)
		dst[ind+BLOCKSIZE2] = src[index[ind+BLOCKSIZE2]+(x+BLOCKSIZE2)*BASESNUM+y*BASESNUM*BASESNUM];

}

__global__ void globalNormalizeProb(float* nodeProb)
{
	const unsigned int y = blockIdx.y;
	const unsigned int x = threadIdx.x;
	const unsigned int ind = y*BASESNUM + x;

	__shared__ float sprob[BASESNUM];

	sprob[x] = nodeProb[ind];
	
	if(x<BASESNUM-BLOCKSIZE3)
		sprob[x+BLOCKSIZE3] = nodeProb[ind+BLOCKSIZE3];
	deviceNormalizeProb(nodeProb,sprob,x,y);
}


//results in weighted coeffs[100,nbases]
__global__ void globalUpdateStep1(float* w_coeff1,float* w_coeff2,int*winner,float*coef1,float*coef2 , float*proj, float sigma_h)
{
	//Threads per block should be larger than the number of subspaces
	const unsigned int y = blockIdx.y;
	const unsigned int x = threadIdx.x;
	const unsigned int ind = y*AGENTSNUM + x;
	const unsigned int ind2 = y+x*BASESNUM;

	if(x<AGENTSNUM)
	{
		short w_row= winner[x]%TOPO_SUBSPACE;
		short w_col= winner[x]/TOPO_SUBSPACE;

		short my_row = y%TOPO_SUBSPACE;
		short my_col = y/TOPO_SUBSPACE;

		float x_s = pow(float(w_row-my_row),2);
		float y_s = pow(float(w_col-my_col),2);

		float func_h = exp( -1.0*(x_s+y_s)/(2*pow(sigma_h,2)) );
		float n_const = func_h/(sqrt(proj[ind2]+EPS));
	
		w_coeff1[ind] = n_const*coef1[ind];//weighted coefficients
		w_coeff2[ind] = n_const*coef2[ind];
		//temp[ind] = n_const;
	}
	__syncthreads();
}


//mat1->X(D,100) mat2->WCof(100,Nb) new, returns weightedInput[D,nbases]
/*
__global__ void globalBlockMulAdd(float*dst,float* data, float*wcoeff )
{
	const unsigned int y = blockIdx.y;
	const unsigned int x = threadIdx.x;
	

	__shared__ float sdatCom[AGENTSNUM];
	__shared__ float sdatRow[AGENTSNUM];
	volatile float* vsdata = sdatRow;
	float rdata;
	int i=0;

	if(x<AGENTSNUM)
	{
		sdatCom[x] = data[x*BASISDIM+y];
	}

	for(i=0; i<BASESNUM; i++)
	{
		if(x<AGENTSNUM)
		{
			sdatRow[x] = wcoeff[x+AGENTSNUM*i];
		}
		__syncthreads();

		if(x<AGENTSNUM)
		{
			sdatRow[x]= rdata = sdatRow[x]*sdatCom[x];
		}
		__syncthreads();
	
		if(x<64)
		{
			sdatRow[x] = rdata = rdata + sdatRow[x+64];
		}
		__syncthreads();
		if(x<32)
		{
	#pragma unroll
			for(int s = 32; s > 0; s >>= 1)
				vsdata[x] = rdata = rdata + vsdata[x+s];
		}
		
		if(x==0)
			dst[y+BASISDIM*i] = vsdata[0]; //contains the sum [D,Nb] at the moment and can change if needed
		__syncthreads();	

	}

	

}*/

//calculate 3 dot products and summ them across the agentsnum

__global__ void globalDotSum(float*dst, float*coeff1, float* coeff2, float* wcoeff1, float* wcoeff2)
{
	const unsigned int y = blockIdx.y;
	const unsigned int x = threadIdx.x;
	const unsigned int ind1 = y*AGENTSNUM + x;

	__shared__  float dat1[AGENTSNUM];
	__shared__  float dat2[AGENTSNUM];
	__shared__  float wdat1[AGENTSNUM];
	__shared__  float wdat2[AGENTSNUM];
	
	dat1[x] = coeff1[ind1];
	dat2[x] = coeff2[ind1];
	wdat1[x] = wcoeff1[ind1];
	wdat2[x] = wcoeff2[ind1];
	
	if(x<(AGENTSNUM-64))
	{
		dat1[x+64] = coeff1[ind1+64];
		dat2[x+64] = coeff2[ind1+64];
		wdat1[x+64] = wcoeff1[ind1+64];
		wdat2[x+64] = wcoeff2[ind1+64];
	}


	__syncthreads();
	
	//deviceMulSum(dst,dat1,wdat2,x,y,0);
	deviceMulSum(dst,dat1,wdat1,x,y,0);
	deviceMulSum(dst,dat2,wdat1,x,y,1);
	deviceMulSum(dst,dat2,wdat2,x,y,2);

	

}

//<200 threads, nb blocks>
__global__ void globalCalcUpdate(float *bases1,float *bases2, float* interSum, float* winput1, float* winput2,float alpha )
{
	const unsigned int y = blockIdx.y;
	const unsigned int x = threadIdx.x;
	const unsigned int ind = y*BASISDIM + x;

	__shared__ float dbases1[BASISDIM];
	__shared__ float dbases2[BASISDIM];
	__shared__ float data1[BASISDIM];
	__shared__ float data2[BASISDIM];
	__shared__ float dwsum[3];

	if(x==0)
	{
		dwsum[0]  = interSum[y+0*BASESNUM]; //11
		dwsum[1]  = interSum[y+1*BASESNUM]; //12
		dwsum[2] =  interSum[y+2*BASESNUM]; //22
	}


	if(x<BASISDIM)
	{
		dbases1[x] = bases1[ind];
		dbases2[x] = bases2[ind];
	}
	__syncthreads();

	if(x<BASISDIM)
	{
		data1[x] = dbases1[x]*dwsum[0] +dbases2[x]*dwsum[1];
		data2[x] = dbases1[x]*dwsum[1] +dbases2[x]*dwsum[2];

		data1[x] = winput1[ind] - data1[x];
		data2[x] = winput2[ind] - data2[x];

	}
	__syncthreads();
	
	if(x<BASISDIM)
	{
		bases1[ind] = dbases1[x] +alpha*data1[x];
		bases2[ind] = dbases2[x] +alpha*data2[x];
	}
	__syncthreads();
}

//input-<[nb,100], nb threads and 100 blocks
__global__ void globalFindMax(int*maxIndex, float*data)
{
	const  int y = blockIdx.y;
	const  int x = threadIdx.x;
	const  int ind = y*BASESNUM + x;
	float rdata;
	float rdata2;


	__shared__   float sdata[BASESNUM];
	__shared__   float sodata[BASESNUM];
	volatile float *vsdata = sdata; 

	if(x<BASESNUM)
	{
		sdata[x] = sodata[x]=rdata= data[ind];
	}
	
	__syncthreads();

	if(x<(BASESNUM-256))
		{
			rdata2 = sdata[x+256];
			sdata[x] = rdata = cumax(rdata,rdata2);

		}
	__syncthreads();

	if(x<128)
		{
			rdata2 = sdata[x+128];
			sdata[x] = rdata = cumax(rdata,rdata2);

		}
	__syncthreads();

	if(x<64)
		{
			rdata2 = sdata[x+64];
			sdata[x] = rdata = cumax(rdata,rdata2);

		}
	__syncthreads();

	if(x<32)
	{	
	#pragma unroll
		for(int s = 32; s > 0; s >>= 1)
		{	
			rdata2 = vsdata[x+s];
			vsdata[x] = rdata = cumax(rdata,rdata2);	
		}
	}
	__syncthreads();
	if(x<BASESNUM)
	{
		if(sodata[x]==sdata[0]) maxIndex[y] =x;
	}


	__syncthreads();	

}

__device__ void deviceMulSum(float* dst,  volatile float* svec1, volatile float* svec2,const unsigned int& x,const unsigned int& y, const unsigned int& z  )
{
	__shared__    float sdata[AGENTSNUM];
	volatile float* vsdata =sdata;

	float rdata = svec1[x]*svec2[x];;
	float rdata_p = x<(AGENTSNUM-64)?svec1[x+64]*svec2[x+64]:0;
	
	sdata[x] = rdata = rdata + rdata_p;

	__syncthreads();

	if(x<32)
		{
			#pragma unroll
			for(int s = 32; s > 0; s >>= 1)
				vsdata[x] = rdata = rdata + vsdata[x+s];
		}
		__syncthreads();
	if(x==0){
		dst[y+BASESNUM*z] = rdata; //
	}
	__syncthreads();	
}

__device__ void deviceNormalizeBases(float* bases,volatile float * sbases,const unsigned int& x, const unsigned int& y)
{
	__shared__ float sdata[BASISDIM];
	volatile float* vsdata = sdata;
	
	__syncthreads();
	float rdata = pow(sbases[x],2);
	float rdata_p = x<(BASISDIM-128)?pow(sbases[x + 128],2):0;
	sdata[x] = rdata = rdata + rdata_p;
	__syncthreads();
	if(x<64)
	{
		sdata[x] = rdata = rdata + sdata[x+64];
	}
	__syncthreads();
	if(x<32)
	{
#pragma unroll
		for(int s = 32; s > 0; s >>= 1)
			vsdata[x] = rdata = rdata + vsdata[x+s];
	}
	if(x==0)
		sdata[0] = pow(rdata,float(0.5));
	__syncthreads();
	const unsigned int ind = y*BASISDIM + x;
	sbases[x]=bases[ind] =  sbases[x]/vsdata[0];
	if(x<BASISDIM-BLOCKSIZE)
		sbases[x+BLOCKSIZE]=bases[ind+BLOCKSIZE] =  sbases[x+BLOCKSIZE]/vsdata[0];
	__syncthreads();
}



__device__ void deviceDotProject(const float* sbases1,volatile float* sbases2,const unsigned int& x)
{
	__shared__ float sdata[BASISDIM];
	volatile float* vsdata = sdata;

	__syncthreads();
	float rdata = sbases1[x]*sbases2[x];
	float rdata_p = x<(BASISDIM-128)?sbases1[x + 128]*sbases2[x + 128]:0;
	sdata[x] = rdata = rdata + rdata_p;
	__syncthreads();
	
	if(x<64)
	{
		sdata[x] = rdata = rdata + sdata[x+64];
	}
	__syncthreads();
	if(x<32)
	{
#pragma unroll
		for(int s = 32; s > 0; s >>= 1)
			vsdata[x] = rdata = rdata + vsdata[x+s];
	}
	if(x==0)
		sdata[0] = rdata; //contains the sum
	__syncthreads();
	

	sbases2[x] = sbases2[x] - sbases1[x]*sdata[0];
	if(x<BASISDIM-BLOCKSIZE)
		sbases2[x+BLOCKSIZE]= sbases2[x+BLOCKSIZE] - sbases1[x+BLOCKSIZE]*sdata[0];
	__syncthreads();

}

__device__ void deviceNormalizeProb(float* nodeProb, const float* sprob, const unsigned int& x, const unsigned int& y)
{
	__shared__ float sdata[BASESNUM];
	volatile float* vsdata = sdata;
	
	__syncthreads();
	float rdata = sprob[x];
	float rdata_p = x<(BASESNUM-256)?sprob[x + 256]:0;
	sdata[x] = rdata = rdata + rdata_p;
	__syncthreads();
	
	if(x<128)
	{
		sdata[x] = rdata = rdata + sdata[x+128];
	}
	__syncthreads();

	if(x<64)
	{
		sdata[x] = rdata = rdata + sdata[x+64];
	}
	__syncthreads();
	if(x<32)
	{
#pragma unroll
		for(int s = 32; s > 0; s >>= 1)
			vsdata[x] = rdata = rdata + vsdata[x+s];
	}
	if(x==0)
		sdata[0] = rdata;
	__syncthreads();
	const unsigned int ind = y*BASESNUM + x;
	
	nodeProb[ind] =  sprob[x]/vsdata[0];
	if(x<BASESNUM-BLOCKSIZE3)
		nodeProb[ind+BLOCKSIZE3] =  sprob[x+BLOCKSIZE3]/vsdata[0];
}



// Host side code

void AssomOnline::cudaNormalizeBases()
{
	globalNormalizeBases<<<dim3(1,BASESNUM),dim3(BLOCKSIZE,1)>>>(bases1,bases2);	
}

void AssomOnline::cudaNormalizeProb()
{
	globalNormalizeProb<<<dim3(1,AGENTSNUM),dim3(BLOCKSIZE3,1)>>>(nodeProb);	
	cudaDeviceSynchronize();
}

void AssomOnline::cudaAssomEncode()
{
	cublasStatus_t stat;
	//float tmpB[BASESNUM*BASISDIM];
/*	float *t1;
	t1=(float*)malloc(BASISDIM*BASESNUM*sizeof(float));
	cudaMemcpy(t1,bases2,sizeof(float)*BASISDIM*BASESNUM,cudaMemcpyDeviceToHost);*/
	//Todo: transpose coeff, but keep the proj matrix in the same way. Can do by cganging the square add functions.  
//	tomat::push(residue,BASISDIM,100,"X",1,0);

	//project
	cublasSgemm(cuhandles[0],CUBLAS_OP_T,CUBLAS_OP_N,AGENTSNUM,BASESNUM,BASISDIM,&cublasOne,residue,BASISDIM,bases1,BASISDIM,&cublasZero,corrBX1,AGENTSNUM);
	cublasSgemm(cuhandles[1],CUBLAS_OP_T,CUBLAS_OP_N,AGENTSNUM,BASESNUM,BASISDIM,&cublasOne,residue,BASISDIM,bases2,BASISDIM,&cublasZero,corrBX2,AGENTSNUM);
	

	//tomat::push(residue,BASISDIM,AGENTSNUM,"X",1,0);
//	tomat::push(bases2,BASISDIM,BASESNUM,"basis2",1,0);


	//cudaMemcpy(tmpB,bases1,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyDeviceToHost);
/*	if(_isnan(tmpB[0]))
	{
		tomat::push(bases1,BASISDIM,BASESNUM,"bases1",1,0);
		std::cout<<"Error! NAA detected\n";
		
	}*/
	
	/*
	//tomat::push(corrBX2,100,BASESNUM,"corr2",1,0);
    tomat::push(bases1,BASISDIM,BASESNUM,"basis1",1,0);
	tomat::push(residue,BASISDIM,100,"X",1,0);
	tomat::push(bases1,BASISDIM*BASESNUM,1,"coef1",1,0);
	tomat::push(corrBX1,100,BASESNUM,"coef",1,0);
	//tomat::push(residue,BASISDIM,100,"X",1,0);
	cudaMemcpy(t1,bases2,sizeof(float)*BASISDIM*BASESNUM,cudaMemcpyDeviceToHost);
	*/

	//square;
	stat=cublasSdgmm(cuhandles[0],CUBLAS_SIDE_LEFT,BASESNUM*AGENTSNUM,1,corrBX1,BASESNUM*AGENTSNUM,corrBX1,1,corrTmp1,BASESNUM*AGENTSNUM);
	stat=cublasSdgmm(cuhandles[1],CUBLAS_SIDE_LEFT,BASESNUM*AGENTSNUM,1,corrBX2,BASESNUM*AGENTSNUM,corrBX2,1,corrTmp2,BASESNUM*AGENTSNUM); 


	cudaDeviceSynchronize();
	//tomat::push(nodeProb,BASESNUM,100,"np0",1,0);
	//proj[]
	stat=cublasSgeam(cuhandles[0], CUBLAS_OP_T,CUBLAS_OP_T,BASESNUM,AGENTSNUM,&cublasOne,corrTmp1,AGENTSNUM,&cublasOne, corrTmp2,AGENTSNUM,proj,BASESNUM);
	
	cudaDeviceSynchronize();
//	tomat::push(corrBX1,100,BASESNUM,"coef1",1,0);
//	tomat::push(corrBX2,100,BASESNUM,"coef2",1,0);
	//tomat::push(proj,BASESNUM,100,"proj",1,0);
	//tomat::push(bases1,BASISDIM,BASESNUM,"basis1",1,0);
	//tomat::push(bases2,BASISDIM,BASESNUM,"basis2",1,0);
	//tomat::push(residue,BASISDIM,100,"X",1,0);
	
	//get emissions
	globalGetEmission<<<dim3(1,AGENTSNUM),dim3(BLOCKSIZE4,1) >>>(emission, proj,normSigmaW, normSigmaN);
	
	//nodeprob update
	stat=cublasSgemm(cuhandles[0],CUBLAS_OP_T,CUBLAS_OP_N,BASESNUM,AGENTSNUM,BASESNUM,&cublasOne,transProb,BASESNUM,nodeProb,BASESNUM,&cublasZero,corrTmp1,BASESNUM);
	stat=cublasSdgmm(cuhandles[0], CUBLAS_SIDE_LEFT, AGENTSNUM*BASESNUM,1,emission,AGENTSNUM*BASESNUM,corrTmp1,1,nodeProb, AGENTSNUM*BASESNUM );
	cudaDeviceSynchronize();

	//tomat::push(corrBX1,100,BASESNUM,"coef1",1,0);
	//tomat::push(corrBX2,100,BASESNUM,"coef2",1,0);
	//tomat::push(transProb,BASESNUM,BASESNUM,"tp",1,0);
	//tomat::push(nodeProb,BASESNUM,100,"np",1,0);
	//tomat::push(proj,BASESNUM,100,"proj",1,0);
	//tomat::push(emission,BASESNUM,100,"emiss",1,0);
	
	//tomat::push(corrTmp1,BASESNUM,100,"sc1",1,0);
	//tomat::push(proj,BASESNUM,100,"proj",1,0);


	cudaNormalizeProb();
	cudaDeviceSynchronize();
	//tomat::push(nodeProb,BASESNUM,100,"npn",1,0);
	//globalFindMax<<<dim3(1,100),dim3(BLOCKSIZE4,1)>>>(winners, nodeProb); //winners based on transitions
	globalFindMax<<<dim3(1,AGENTSNUM),dim3(BLOCKSIZE4,1)>>>(winners, proj); //winners based on max Proj
	cudaDeviceSynchronize();
	
	//tomat::push(nodeProb,BASESNUM,100,"prob",1,0);



	//globalGetWinProj<<<dim3(1,1),dim3(BLOCKSIZE2,1) >>> (proj, coef, winErr, winners); //maybe a more efficient method?
	//globalGetWinErr<<<dim3(1,1),dim3(BLOCKSIZE2,1) >>> (proj, winErr, winners); //this is the original
	globalGetWinErrProj<<<dim3(1,1),dim3(BLOCKSIZE2,1) >>> (proj, winErr,winProj,winners);
	
	float ratioA = 0.01;  
	cublasStatus_t stats = cublasSgemm(cuhandles[0],CUBLAS_OP_N, CUBLAS_OP_N, BASESNUM, 1, AGENTSNUM,&ratioA, proj, BASESNUM ,ones, AGENTSNUM, &cublasZero, coef, BASESNUM);
	
	cudaDeviceSynchronize();
	globalFindMax<<<dim3(1,100),dim3(BLOCKSIZE4,1)>>>(winners, nodeProb); //replace winners based on alpha matrix for the ASSOM update
//	tomat::push(winners,100,1,"winners",1,0);
//	tomat::push(winErr,100,1,"winErr",1,0);
	//tomat::push(nodeProb,BASESNUM,100,"np",1,0);
	//tomat::push(proj,BASESNUM,100,"proj",1,0);
	//tomat::push(winners,100,1,"w",1,0);
	//tomat::push(winErr,AGENTSNUM,1,"winerr",1,0);
	//tomat::push(coef,BASESNUM+1,1,"coef",1,0);

	//coef and error for the winners calculated. Coef [nbasis+1], and winErr[100]. The error should be averaged to form the reward. The feature vector is already formed.
	

}


void AssomOnline::cudaUpdateBases()
{
	float alpha = ALPHA_A*exp(-1*float(iter)/TCONST)+ALPHA_C;
	float sigma_h = (SIGMA_A/SIGMA_B)*exp(-1*float(iter)/TCONST)+SIGMA_C;
	//float sigma_h = SIGMA_A/(SIGMA_B+iter)+SIGMA_C;
	//cublasStatus_t stat;
	///float tmpB1[BASISDIM*BASESNUM];
	//float tmpB2[BASISDIM*BASESNUM];

	//considering whether we could use the same containers for coefficients 
	globalUpdateStep1<<<dim3(1,BASESNUM),dim3(128,1) >>>(corrTmp1,corrTmp2,winners,corrBX1,corrBX2,proj, sigma_h); //wcoeff[100,nb]
	cudaDeviceSynchronize();

	//tomat::push(corrBX1,100,BASESNUM,"coef1",1,0);
	//tomat::push(corrBX2,100,BASESNUM,"coef2",1,0);
	//tomat::push(corrTmp1,100,BASESNUM,"wc1",1,0);
	//tomat::push(corrTmp2,100,BASESNUM,"wc2",1,0);
	
	//tomat::push(winners,100,1,"win",1,0);
	//tomat::push(proj,BASESNUM,100,"proj",1,0);
	
	//mat1->X(D,100) mat2->WCof(Nb,100)
	cublasSgemm(cuhandles[0],CUBLAS_OP_N,CUBLAS_OP_N,BASISDIM,BASESNUM,AGENTSNUM,&cublasOne,residue,BASISDIM,corrTmp1,AGENTSNUM,&cublasZero,wInputTmp3,BASISDIM);
	cublasSgemm(cuhandles[1],CUBLAS_OP_N,CUBLAS_OP_N,BASISDIM,BASESNUM,AGENTSNUM,&cublasOne,residue,BASISDIM,corrTmp2,AGENTSNUM,&cublasZero,wInputTmp4,BASISDIM);
	cudaDeviceSynchronize();

	//tomat::push(residue,200,100,"X",1,0);
//	tomat::push(wInputTmp3,200,BASESNUM,"wi1",1,0);
	//tomat::push(wInputTmp4,200,BASESNUM,"wi2",1,0);
	

	//wInput1 1,2,3 -> 11,12,22 vec[nb]
	globalDotSum<<<dim3(1,BASESNUM), dim3(64,1)>>>(wInputTmp1, corrBX1, corrBX2, corrTmp1, corrTmp2);
	cudaDeviceSynchronize();
	//cudaError_t error = cudaGetLastError();
	//wInput2 1,2,3,4 -> cc and bases mul
	//tomat::push(bases1,BASISDIM,BASESNUM,"bases1",1,0);
	//tomat::push(bases2,BASISDIM,BASESNUM,"bases2",1,0);
	
	globalCalcUpdate<<<dim3(1,BASESNUM), dim3(224,1)>>>(bases1,bases2, wInputTmp1, wInputTmp3, wInputTmp4,alpha );
	
	cudaDeviceSynchronize();
	//tomat::push(bases1,BASISDIM,BASESNUM,"bases1_2",1,0);
	//tomat::push(bases2,BASISDIM,BASESNUM,"bases2_2",1,0);
	
	cudaNormalizeBases();
	//cudaMemcpy(bases2,tmpBases2,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyDeviceToDevice);
	

	//cudaDeviceSynchronize();
//	cudaMemcpy(tmpB1,bases1,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyDeviceToHost);
//	cudaMemcpy(tmpB2,bases2,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyDeviceToHost);
	
/*	for(int i=0; i<BASISDIM*BASESNUM; i++)
	{
		if(_isnan(tmpB1[i])||_isnan(tmpB2[i]))
		{
		
			tomat::push(residue,BASISDIM,100,"X",1,0);
			tomat::push(corrBX1,AGENTSNUM,BASESNUM,"X1",1,0);
			tomat::push(corrBX2,AGENTSNUM,BASESNUM,"X2",1,0);
			tomat::push(wInputTmp4,BASISDIM,BASESNUM,"X3",1,0);
			tomat::push(wInputTmp3,BASISDIM,BASESNUM,"X4",1,0);
			tomat::push(wInputTmp1,BASESNUM,3,"X5",1,0);
			tomat::push(corrTmp1,AGENTSNUM,BASESNUM,"X6",1,0);
			tomat::push(corrTmp2,AGENTSNUM,BASESNUM,"X7",1,0);
			tomat::push(bases1,BASISDIM,BASESNUM,"bases1_3",1,0);
			tomat::push(bases2,BASISDIM,BASESNUM,"bases2_3",1,0);
			
			tomat::push(proj,BASESNUM,AGENTSNUM,"X8",1,0);
			tomat::push(wInputTmp2,AGENTSNUM,BASESNUM,"X9",1,0);
			std::cout<<"Error! NAA detected\n";
		}
	}*/


	
	
	
	iter = iter+1;
}