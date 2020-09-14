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


#define TESTNUM 25
#define FILT_RANGE 41

class ImageLoader
{
private:
	vector<Mat> imageData;
	int image_width;
	int image_height;
	int window_size;

	Mat leftImg,rightImg;

	int tilt_action;
	int version_angle;   //left--negative;  right--positive
	int vergence_angle;  //inner--positive; outter--negative
	int input_disparity;

	int inner_offset;
	int inner_limit;
	

	float* disparity_pdf;
	int probRange;
	
	int image_index;
	//set frame number
	int curFrame;
	int frameNum;
	int curEnv;
	int goal;
	int speed;

	float* csmapCell;
	float* dispMap;
	float* dispMapPool;
	float* localMaxCell;

	float* new_csmap;
	
	float* discount_filter;

	Mat leftWtmp,rightWtmp;
	


public:
	float dispDiff;

	int left_x_pos;
	int right_x_pos;
	int y_pos;
	
	Mat leftWin,rightWin;
	

	ImageLoader();
	~ImageLoader();

	void show_image_h_mono();
	void show_image_m_mono();

	void show_image_h();
	void show_image_m();
	void show_image_h(int flag);
	void show_image_m(int flag);

	void show_image_h(int flag, int option);

	void show_image(Mat left, Mat right);
	void show_image(Mat left);

	//for joint learning
	void add_jitter();

	void get_fixation_point(int index);
	void get_fixation_point(int index, float randPercent);
	void get_fixation_point2(int index, float randPercent);

	void load_image();
	void get_frame_number_saliency(int i, int interval);

	void get_frame_number_together(int frameIndex, int lx, int ly, int loadFlag);
	void get_frame_number_together(int frameIndex, float randPercent, int loadFlag);
	void get_frame_number_together2(int frameIndex, float randPercent, int loadFlag);

	float get_dispMap_xy(int y, int x);

	float get_disp();
	float get_disp2();

	void get_image_input1(int frameIndex, int lx, int ly, int action_taken);
	void get_image_input1(int frameIndex, int lx, int ly, int action_taken, int disp, int flag);


	void get_fixation_point_saliency(int frameIndex, int flag, float randPercent);
	void get_fixation_point_max(int frameIndex, int flag);
	void get_fixation_point_max2(int frameIndex, int reload);
	void get_image_input2(int frameIndex, int action_taken, int disp, int flag);

	void window_position_after_action_stereo(int action);
	void window_position_after_action_stereo(int dis, int iter, int action, int interval);

	void window_position_after_action_mono(int dis);
	void window_position_after_action_mono(int dis, int iter, int action, int interval);

	float* getLeftWindow();
	float* getRightWindow();


	float* getFilter();
};
