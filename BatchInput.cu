#include "BatchInput.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



__device__ void zeroMeanBatchs(float * sbatchInput, const unsigned int& y)
{
	__shared__ float sdata[128];
	volatile float* vsdata = sdata;
	float mean;
	
	__syncthreads();
	float rdata = sbatchInput[y];
	float rdata_p = y<PATCHWIDTH*PATCHWIDTH-64?sbatchInput[y + 64]:0;
	float rdata64 = sbatchInput[y+128];
	float rdata_p64 = y<PATCHWIDTH*PATCHWIDTH-64?sbatchInput[y + 128 + 64]:0;

	if(y<64)
	{
		sdata[y] = rdata = rdata + rdata_p;
		sdata[y+64] = rdata64 = rdata64 + rdata_p64;
	}
	__syncthreads();
	if(y<32)
	{
#pragma unroll
		for(int s = 32; s > 0; s >>= 1)
		{
			if(y<s)
			{
				vsdata[y] = rdata = rdata + vsdata[y+s];
				vsdata[y+64] = rdata64 = rdata64 + vsdata[y+64+s];
			}
		}
	}
	if(y==0)
	{
		sdata[y] = rdata/float(PATCHWIDTH*PATCHWIDTH);
		sdata[y+64] = rdata64/float(PATCHWIDTH*PATCHWIDTH);
	}
	__syncthreads();
	sbatchInput[y] =  sbatchInput[y]-vsdata[0];
	sbatchInput[y+128] =  sbatchInput[y+128]-vsdata[64];
}

__device__ void deviceNormalizeBatchs(float* batchInput,float const* sbatchInput,const unsigned int& x, const unsigned int& y)
{
	__shared__ float sdata[128];
	volatile float* vsdata = sdata;
	float sumsq = 0;
	
	__syncthreads();
	float rdata = pow(sbatchInput[y],2);
	float rdata_p = y<PATCHWIDTH*PATCHWIDTH-64?pow(sbatchInput[y + 64],2):0;
	float rdata64 = pow(sbatchInput[y+128],2);
	float rdata_p64 = y<PATCHWIDTH*PATCHWIDTH-64?pow(sbatchInput[y + 128 + 64],2):0;
	if(y<64)
	{
		sdata[y] = rdata = rdata + rdata_p;
		sdata[y+64] = rdata64 = rdata64 + rdata_p64;
	}
	__syncthreads();
	if(y<32)
	{
#pragma unroll
		for(int s = 32; s > 0; s >>= 1)
		{
			vsdata[y] = rdata = rdata + vsdata[y+s];
			vsdata[y+64] = rdata64 = rdata64 + vsdata[y+64+s];
		}
	}

#if COM1NORM

	if(y==0)
	{
		sdata[y] = pow(rdata+rdata64+float(EPS),float(0.5));
	//	sdata[y+64] = pow(rdata64+float(EPS),float(0.5));
	}

	__syncthreads();


	const unsigned int ind = y + x*BASISDIM;

	batchInput[ind] =  sbatchInput[y]/vsdata[0];
	batchInput[ind+PATCHWIDTH*PATCHWIDTH] =  sbatchInput[y+128]/vsdata[0];
	if(vsdata[0]<1e-3)
	{
	//	printf("\nsbsb\n");
		batchInput[ind] = batchInput[ind+PATCHWIDTH*PATCHWIDTH] = sqrt(0.005);
	}

//	batchInput[ind] =  sbatchInput[y];
//	batchInput[ind+PATCHWIDTH*PATCHWIDTH] =  sbatchInput[y+128];

#else

	if(y==0)
	{
		sdata[y] = pow(rdata+float(EPS),float(0.5));
		sdata[y+64] = pow(rdata64+float(EPS),float(0.5));
	}

	__syncthreads();

	const unsigned int ind = y + x*BASISDIM;

	batchInput[ind] =  sbatchInput[y]/vsdata[0];
	batchInput[ind+PATCHWIDTH*PATCHWIDTH] =  sbatchInput[y+128]/vsdata[64];
	if(vsdata[0]<1e-3)
		batchInput[ind] = float(1.0)/float(PATCHWIDTH);
	if(vsdata[64]<1e-3)
		batchInput[ind+PATCHWIDTH*PATCHWIDTH] = float(1.0)/float(PATCHWIDTH);
#endif
	
	

}

__global__ void globalCuts(float * lF,float* rF, float* bI, int dSfS, int ps)
{
	const unsigned int x = blockIdx.y*ps + threadIdx.y;
	const unsigned int y = blockIdx.x*ps + threadIdx.x;

	const unsigned int batchx = blockIdx.x + blockIdx.y*ROWPATCHNUM;
	const unsigned int batchy = threadIdx.x+ threadIdx.y*PATCHWIDTH;
	
	__shared__ float sbatchInput[256];

	sbatchInput[batchy] = lF[y*dSfS+x];
	sbatchInput[batchy+128] = rF[y*dSfS+x];

	zeroMeanBatchs(sbatchInput,batchy);
	deviceNormalizeBatchs(bI,sbatchInput,batchx,batchy);

}




void BatchInput::GpuCut()
{

	globalCuts<<<dim3(ROWPATCHNUM,ROWPATCHNUM),dim3(PATCHWIDTH,PATCHWIDTH)>>>(leftFoveaWh,rightFoveaWh,batchInput,downSampled_foveaSize,patchShift);
	cudaThreadSynchronize();

}

 __global__ void real2complex (float *a, cufftComplex *c, int N) 
{ 
	
	int idx = blockIdx.x*blockDim.x+threadIdx.x; 
	int idy = blockIdx.y*blockDim.y+threadIdx.y; 
	if ( idx < N && idy <N) 
	{ 
	 int index = idx + idy*N; 
	 c[index].x = a[index]; 
	 c[index].y = 0.f; 
	} 
	__syncthreads();
}

 /*compute idx and idy, the location of the element in the original NxN array*/ 
 __global__ void complex2real_scaled (cufftComplex *c, float *a, int M, int N, float scale) 
{ 

	int idx = blockIdx.x*blockDim.x+threadIdx.x; 
	int idy = blockIdx.y*blockDim.y+threadIdx.y; 
	if ( (idx>=(M-N)/2)&&(idx<M-(M-N)/2) && (idy>=(M-N)/2)&&(idy<M-(M-N)/2)) 
	 { 
		int index = idx + idy*M; 
		int index2 = idx-(M-N)/2 + (idy-(M-N)/2)*N; 
		a[index2] = scale*c[index].x ; 
	 }
	__syncthreads();
} 

inline __device__ void mulAndScale(cuComplex& a, const cuComplex& b, const float& c){
     cuComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
     a = t;
 }

 __global__ void modulateAndNormalize_kernel(cuComplex *d_Dst, cuComplex *d_Src, int N, float c )
 {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; 
	int idy = blockIdx.y*blockDim.y+threadIdx.y; 
	cuComplex a,b;
	
	if ( idx < N && idy <N) 
	{ 
	 int i = idx + idy*N; 
		 a = d_Src[i];
		 b = d_Dst[i];
		 mulAndScale(a, b, c);
		 d_Dst[i] = a;
	}
	__syncthreads();
 }

void BatchInput::imgWhitening()
{

	real2complex<<< dim3(16,16), dim3(16,16)>>>(leftFovea, d_dataL, FILT_WIDTH);
	real2complex<<< dim3(16,16), dim3(16,16)>>>(rightFovea, d_dataR, FILT_WIDTH);
	cudaThreadSynchronize();

	cufftExecC2C(fftPlan, (cufftComplex *)d_dataL, (cufftComplex *)d_DataSpectrumL,CUFFT_FORWARD );
	cufftExecC2C(fftPlan, (cufftComplex *)d_dataR, (cufftComplex *)d_DataSpectrumR,CUFFT_FORWARD );


	modulateAndNormalize_kernel<<<dim3(16,16), dim3(16,16)>>> (d_DataSpectrumL, d_filter, FILT_WIDTH , 1);
	modulateAndNormalize_kernel<<<dim3(16,16), dim3(16,16)>>> (d_DataSpectrumR, d_filter, FILT_WIDTH , 1);
	cudaThreadSynchronize();


	cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrumL, (cufftComplex *)d_resultL,CUFFT_INVERSE );
	cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrumR, (cufftComplex *)d_resultR,CUFFT_INVERSE );

	complex2real_scaled<<< dim3(16,16), dim3(16,16)>>>(d_resultL, leftFoveaWh, FILT_WIDTH, downSampled_foveaSize, float(1)/(FILT_WIDTH*FILT_WIDTH) );
	complex2real_scaled<<< dim3(16,16), dim3(16,16)>>>(d_resultR, rightFoveaWh, FILT_WIDTH, downSampled_foveaSize, float(1)/(FILT_WIDTH*FILT_WIDTH));


}

void BatchInput::initFilt(float* f)
{
	filtdata=f;
	cudaMemcpy(d_filterReal,filtdata,sizeof(float)*FILT_WIDTH*FILT_WIDTH,cudaMemcpyHostToDevice);
	real2complex<<< dim3(16,16), dim3(16,16)>>>(d_filterReal, d_filter, FILT_WIDTH);
}