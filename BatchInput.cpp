#include"BatchInput.h"

BatchInput::BatchInput(int dsRate, int pf, float* f)
{
	windowSize=220;

	downSampling=dsRate;
	patchShift=pf;

	downSampled_foveaSize=10+9*pf;
	foveaSize=dsRate*FILT_WIDTH;
	winPos_offset=(windowSize-foveaSize)/2;

//	printf("downSampled_foveaSize=%d, foveaSize=%d, winPos_offset=%d\n",downSampled_foveaSize, foveaSize, winPos_offset);
//	system("pause");

	cudaMalloc((void**)&leftFovea,sizeof(float)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&rightFovea,sizeof(float)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&batchInput,sizeof(float)*AGENTSNUM*BASISDIM);

	

	cudaMalloc((void**)&leftFovea,sizeof(float)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&rightFovea,sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	cudaMalloc((void**)&leftFoveaWh,sizeof(float)*downSampled_foveaSize*downSampled_foveaSize);
	cudaMalloc((void**)&rightFoveaWh,sizeof(float)*downSampled_foveaSize*downSampled_foveaSize);
	

	cudaMalloc((void**)&d_DataSpectrumL,sizeof(cuComplex)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&d_DataSpectrumR,sizeof(cuComplex)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&d_filter,sizeof(cuComplex)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&d_dataL,sizeof(cuComplex)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&d_dataR,sizeof(cuComplex)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&d_resultR,sizeof(cuComplex)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&d_resultL,sizeof(cuComplex)*FILT_WIDTH*FILT_WIDTH);
	cudaMalloc((void**)&d_filterReal,sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	cufftPlan2d(&fftPlan, FILT_WIDTH, FILT_WIDTH , CUFFT_C2C);

	filtdata=new float [FILT_WIDTH*FILT_WIDTH];
	initFilt(f);
}


BatchInput::~BatchInput()
{
	cudaFree(leftFovea);
	cudaFree(rightFovea);
	cudaFree(leftFoveaWh);
	cudaFree(rightFoveaWh);
	cudaFree(batchInput);

	cudaFree(d_DataSpectrumL);
	cudaFree(d_DataSpectrumR);
	cudaFree(d_filter);
	cudaFree(d_dataL);
	cudaFree(d_dataR);
	cudaFree(d_resultL);
	cudaFree(d_resultR);
	cudaFree(d_filterReal);

	delete [] filtdata;

}


void BatchInput::get_batch_input(Mat leftW, Mat rightW, int left_x_pos, int right_x_pos, int y_pos)
{
	
	leftW.rowRange(winPos_offset,winPos_offset+foveaSize)
		   .colRange(winPos_offset,winPos_offset+foveaSize)
		   .convertTo(leftFtmp,CV_32FC1);

	rightW.rowRange(winPos_offset,winPos_offset+foveaSize)
		   .colRange(winPos_offset,winPos_offset+foveaSize)
		   .convertTo(rightFtmp,CV_32FC1);



	float resizeRate=float(1)/float(downSampling);
	resize(leftFtmp,leftF,Size(0,0),resizeRate,resizeRate,3);
	resize(rightFtmp,rightF,Size(0,0),resizeRate,resizeRate,3);
/*
	imshow( "leftF", leftF );
	waitKey(1);
*/
	cudaMemcpy(leftFovea,leftF.data,sizeof(float)*FILT_WIDTH*FILT_WIDTH,cudaMemcpyHostToDevice);
	cudaMemcpy(rightFovea,rightF.data,sizeof(float)*FILT_WIDTH*FILT_WIDTH,cudaMemcpyHostToDevice);

	imgWhitening();

	GpuCut();
}


float* BatchInput::getLfovea()
{
	return leftFovea;
}

float* BatchInput::getRfovea()
{
	return rightFovea;
}

float* BatchInput::getLfoveaWh()
{
	return leftFoveaWh;
}

float* BatchInput::getRfoveaWh()
{
	return rightFoveaWh;
}

int BatchInput::getFoveaWidth()
{
	return downSampled_foveaSize;
}

float* BatchInput::getBatch()
{
	return batchInput;
}