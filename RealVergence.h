#pragma once

#include "AssomOnline.h"
#include "ReinforcementLearner.h"
#include <cuda_runtime.h>
#include "config.h"
#include "ImageLoader.h"
#include "BatchInput.h"
#include "DataRecord.h"
#include "ToMatlab.h"
#include <windows.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <conio.h>
#define T_TRAIN	int(6E6)


double PCFreq = 0.0;
__int64 CounterStart = 0;
__int64 realCounter = 0;

cublasHandle_t cuhandle;
cublasHandle_t cuhandles[BLASSTREAMS];
cudaStream_t streams[BLASSTREAMS]; 
float cublasOne;
float cublasZero;
float cublasNegOne;
int seed;

int initCuda();
void destroyCuda();

int vergence_command[11]= {-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16};



void trainJoint();
void StartCounter();
double GetCounter();