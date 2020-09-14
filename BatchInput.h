#pragma once
#include <vector>
#include <string>
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <cublas_v2.h>
#include <opencv2\core\core.hpp>
#include <cuda_runtime.h>
#include <math.h>
#include <array>
#include <functional>
#include <cufft.h>
#include <device_launch_parameters.h>
#pragma comment(lib, "cuFFT")

//#include "cuda_bug.h"
#include "config.h"
#include "ToMatlab.h"
using namespace cv;

#define COM1NORM 1


class BatchInput
{
private:

	int windowSize;
	int foveaSize;
	int downSampled_foveaSize;
	int downSampling;
	int winPos_offset;
	int patchShift;

	

	Mat leftFtmp,rightFtmp;
	Mat leftF,rightF;

	float *leftFovea, *rightFovea;
	float *leftFoveaWh, *rightFoveaWh;

	float* batchInput;

	cufftHandle fftPlan;
	cuComplex *d_DataSpectrumL, *d_DataSpectrumR, *d_filter, *d_dataL, *d_dataR, *d_resultL ,*d_resultR;
	float* d_filterReal;
	float* filtdata;

	void imgWhitening();
	void initFilt(float* f);

	void GpuCut();

public:

	BatchInput(int dsRate, int pf, float* f);
	~BatchInput();

	void get_batch_input(Mat imgWL, Mat imgWR, int xL, int xR, int yLR);
	
	float* getLfovea();
	float* getRfovea();

	float* getLfoveaWh();
	float* getRfoveaWh();

	int getFoveaWidth();
	float* getBatch();

};