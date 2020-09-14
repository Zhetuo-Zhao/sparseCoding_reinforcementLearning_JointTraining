#include "ToMatlab.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

Engine *ep;
void tomat::start()
{
	if (!(ep = engOpen("\0"))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return;
	}
	engEvalString(ep,"cd 'D:\\ZHAOYU\\MPil\\Record_by_Date\\2013-02-19_CUDA_HappyNewYear\\matlabProgramForCudaTest';");
}

void tomat::close()
{
	engClose(ep);
}