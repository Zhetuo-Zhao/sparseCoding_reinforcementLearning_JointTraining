#include "ImageLoader.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



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

void ImageLoader::imgWhitening()
{

	real2complex<<< dim3(16,16), dim3(16,16)>>>(leftWindow, d_dataL, FILT_WIDTH);
	real2complex<<< dim3(16,16), dim3(16,16)>>>(rightWindow, d_dataR, FILT_WIDTH);
	cudaThreadSynchronize();

	cufftExecC2C(fftPlan, (cufftComplex *)d_dataL, (cufftComplex *)d_DataSpectrumL,CUFFT_FORWARD );
	cufftExecC2C(fftPlan, (cufftComplex *)d_dataR, (cufftComplex *)d_DataSpectrumR,CUFFT_FORWARD );


	modulateAndNormalize_kernel<<<dim3(16,16), dim3(16,16)>>> (d_DataSpectrumL, d_filter, FILT_WIDTH , 1);
	modulateAndNormalize_kernel<<<dim3(16,16), dim3(16,16)>>> (d_DataSpectrumR, d_filter, FILT_WIDTH , 1);
	cudaThreadSynchronize();


	cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrumL, (cufftComplex *)d_resultL,CUFFT_INVERSE );
	cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrumR, (cufftComplex *)d_resultR,CUFFT_INVERSE );

	complex2real_scaled<<< dim3(16,16), dim3(16,16)>>>(d_resultL, leftWindowWh, FILT_WIDTH, FILT_WIDTH, float(1)/(FILT_WIDTH*FILT_WIDTH) );
	complex2real_scaled<<< dim3(16,16), dim3(16,16)>>>(d_resultR, rightWindowWh, FILT_WIDTH, FILT_WIDTH, float(1)/(FILT_WIDTH*FILT_WIDTH));


}

void ImageLoader::initFilt(float* f)
{
	filtdata=f;
	cudaMemcpy(d_filterReal,filtdata,sizeof(float)*FILT_WIDTH*FILT_WIDTH,cudaMemcpyHostToDevice);
	real2complex<<< dim3(16,16), dim3(16,16)>>>(d_filterReal, d_filter, FILT_WIDTH);
}