
#pragma once


#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <cublas_v2.h>

#define BABY 0

#define FILT_WIDTH 55

const int envSetSize = 250;
const int envDuration = 250;//100;
const int envGroup = 10;
const int totalEnv = 15;

#define CSMAP_WIDTH 168
#define CSMAP_HEIGHT 128

#define CV_TYPE CV_32FC1
#define EPS	1E-16
//#define T_TRAIN	int(2E4) defined in Icub_vergence.h
#define PI 3.14159265358979325
#define CHECKPOINTSNUM 30
#define BLASSTREAMS 2

#define TESTTRAJ 0
#define NOVERGENCE 0

#define TOPO_SUBSPACE 18
#define BASESNUM int(TOPO_SUBSPACE*TOPO_SUBSPACE)
#define BASISDIM 200
#define NUMINUSE 10
#define AGENTSNUM 100
#define ETA float(0.01)
#define ALPHA_A float(0.0050) //0.005
#define ALPHA_C float(0.0002)
#define SIGMA_A float(2.5)
#define SIGMA_B float(1)
#define SIGMA_C float(0.1)
#define SIGMA_N float(0.2) //0.2
#define SIGMA_W float(2) //2
#define INPUTSQNORM 1

#define TCONST float(40000) //float(50000)
#define TCONST2 float(20000)

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480


#define PATCHWIDTH 10
#define PATCHSHIFT 5
#define PATCHNUM AGENTSNUM
#define PUPILDIST 120//300
#define ROWPATCHNUM 10 //sqrt<int>(AGENTSNUM)
#define FOV 5.5/180*PI

#define ACTIONNUM 11
#define SCALENUM 3
#define FEATUREDIM int(1+BASESNUM)

#define ALPHA_V float(0.02) //0.1
#define ALPHA_P float(0.002) //0.002
#define ALPHA_N float(0.04) //0.08

#define ALPHA_V2 float(0.01) //0.1
#define ALPHA_P2 float(0.001) //0.002
#define ALPHA_N2 float(0.02) //0.08

#define ALPHA_V_NEW float(0.1) //0.1
#define ALPHA_P_NEW float(0.01) //0.002
#define ALPHA_N_NEW float(0.1) //0.08

#define GAMMA  float(0.05) //0.05
#define LAMBDA float(0.01)  //(0.01) is good
#define XI float(0.3)
#define INITIALWEIGHTSRANGE1 0.0080
#define INITIALWEIGHTSRANGE2 0.005
#define ACTIONS -5,-4,-3,-2,-1,0,1,2,3,4,5
#define RL_UPDATE_INT 1

#define IMGPLANE float(2) //0.8

#define BLOCKSIZE 192
#define BLOCKSIZE2 224
#define BLOCKSIZE3 256
#define BLOCKSIZE4 400

#define N_IMG_EPOCH 10
extern cublasHandle_t cuhandle;
extern cublasHandle_t cuhandles[];
extern cudaStream_t streams[];
extern float cublasOne;
extern float cublasZero;
extern float cublasNegOne;
using namespace cv;

#pragma comment(lib, "cudart")
#pragma comment(lib, "cublas")
//#pragma comment(lib, "curand")
#pragma comment(lib, "libeng")
#pragma comment(lib, "libmwlapack")
#pragma comment(lib, "libmex")
#pragma comment(lib, "libmx")


#pragma comment(lib, "opencv_core249d")
#pragma comment(lib, "opencv_highgui249d")
#pragma comment(lib, "opencv_imgproc249d")

