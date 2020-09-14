#include"ImageLoader.h"

ImageLoader::ImageLoader()
{
	image_width=IMAGE_WIDTH;
	image_height=IMAGE_HEIGHT;
	window_size=220;//FOVEAWIDTH;

	version_angle=0;
	vergence_angle=0;
	input_disparity=0;
	
	image_index=0;
	curFrame=0;
	frameNum=0;
	curEnv=0;
	goal=0;
	speed=0;

	left_x_pos=(image_width-window_size)/2;
	right_x_pos=(image_width-window_size)/2;
	y_pos=(image_height-window_size)/2;
	
	inner_limit=40;

	csmapCell=new float [CSMAP_WIDTH*CSMAP_HEIGHT*TESTNUM];	
	localMaxCell=new float [CSMAP_WIDTH*CSMAP_HEIGHT*TESTNUM];	

	dispMap=new float [IMAGE_WIDTH*IMAGE_HEIGHT];
	dispMapPool=new float [IMAGE_WIDTH*IMAGE_HEIGHT*TESTNUM];

	new_csmap= new float [CSMAP_WIDTH*CSMAP_HEIGHT];

	discount_filter= new float [FILT_RANGE*FILT_RANGE];

	float errMsum=0;
	for (int iy=0;iy<FILT_RANGE;iy++)
	{
		for (int ix=0;ix<FILT_RANGE;ix++)
		{
			int offset=(FILT_RANGE-1)/2;
			float tmp=-float((ix-offset)*(ix-offset))/float(200)-float((iy-offset)*(iy-offset))/float(200);
			discount_filter[ix*FILT_RANGE+iy]=1-exp(tmp)*0.7;

	//		printf("discount_filter(%d, %d)=%f\n",ix,iy,discount_filter[ix*FILT_RANGE+iy]);
	//		system("pause");
		}
	}

}


ImageLoader::~ImageLoader()
{
	delete [] csmapCell;

	delete [] dispMap;
	delete [] dispMapPool;

	delete [] localMaxCell;
	
	delete [] new_csmap;
	delete [] discount_filter;
}



void ImageLoader::show_image_h_mono()
{


	Mat lh=leftImg.clone();

	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;


	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(lh,l_a1,l_b1,(255,0,0),3);
	rectangle(lh,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(lh,l_a2,l_b2,(255,0,0),3);
	rectangle(lh,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(lh,l_a3,l_b3,(255,0,0),3);
	rectangle(lh,r_a3,r_b3,(255,0,0),3);



	imshow( "leftImg_h", lh );
	waitKey(1);
}

void ImageLoader::show_image_m_mono()
{
	Mat lm=leftImg.clone();

	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;

	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(lm,l_a1,l_b1,(255,0,0),3);
	rectangle(lm,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(lm,l_a2,l_b2,(255,0,0),3);
	rectangle(lm,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(lm,l_a3,l_b3,(255,0,0),3);
	rectangle(lm,r_a3,r_b3,(255,0,0),3);


	imshow( "leftImg_m", lm );
	waitKey(1);

}



void ImageLoader::show_image_h()
{
//	printf("left_x=%d, right_x=%d, left_y=%d, right_y=%d\n",left_x_pos,right_x_pos,y_pos,y_pos);

	Mat lh=leftImg.clone();
	Mat rh=rightImg.clone();

	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;
	Point r_a4, r_b4, l_a4, l_b4;

/*
	r_a4.x=left_x_pos-get_disp2()+82;
	r_a4.y=y_pos+82;
	r_b4.x=left_x_pos-get_disp2()+window_size-83;
	r_b4.y=y_pos+window_size-83;

	rectangle(lh,l_a4,l_b4,(255,255,255),3);
	rectangle(rh,r_a4,r_b4,(255,255,255),3);
*/



	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(lh,l_a1,l_b1,(255,0,0),3);
	rectangle(rh,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(lh,l_a2,l_b2,(255,0,0),3);
	rectangle(rh,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(lh,l_a3,l_b3,(255,0,0),3);
	rectangle(rh,r_a3,r_b3,(255,0,0),3);



	imshow( "leftImg_h", lh );
	waitKey(1);
	imshow( "rightImg_h", rh );
	waitKey(1);
}

void ImageLoader::show_image_m()
{
//	printf("left_x=%d, right_x=%d, left_y=%d, right_y=%d\n",left_x_pos,right_x_pos,y_pos,y_pos);
	Mat lm=leftImg.clone();
	Mat rm=rightImg.clone();

	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;
	Point r_a4, r_b4, l_a4, l_b4;
/*
	r_a4.x=left_x_pos-get_disp2()+82;
	r_a4.y=y_pos+82;
	r_b4.x=left_x_pos-get_disp2()+window_size-83;
	r_b4.y=y_pos+window_size-83;

	rectangle(lm,l_a4,l_b4,(255,255,255),3);
	rectangle(rm,r_a4,r_b4,(255,255,255),3);
*/


	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(lm,l_a1,l_b1,(255,0,0),3);
	rectangle(rm,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(lm,l_a2,l_b2,(255,0,0),3);
	rectangle(rm,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(lm,l_a3,l_b3,(255,0,0),3);
	rectangle(rm,r_a3,r_b3,(255,0,0),3);


	imshow( "leftImg_m", lm );
	waitKey(1);
	imshow( "rightImg_m", rm );
	waitKey(1);
}

void ImageLoader::show_image_h(int flag)
{
//	printf("left_x=%d, right_x=%d, left_y=%d, right_y=%d\n",left_x_pos,right_x_pos,y_pos,y_pos);

	Mat lh=leftImg.clone();
	Mat rh=rightImg.clone();

	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;
	Point r_a4, r_b4, l_a4, l_b4;

	if (flag)
	{
		r_a4.x=left_x_pos-get_disp()+82;
		r_a4.y=y_pos+82;
		r_b4.x=left_x_pos-get_disp()+window_size-83;
		r_b4.y=y_pos+window_size-83;

		rectangle(lh,l_a4,l_b4,(255,255,255),3);
		rectangle(rh,r_a4,r_b4,(255,255,255),3);
	}



	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(lh,l_a1,l_b1,(255,0,0),3);
	rectangle(rh,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(lh,l_a2,l_b2,(255,0,0),3);
	rectangle(rh,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(lh,l_a3,l_b3,(255,0,0),3);
	rectangle(rh,r_a3,r_b3,(255,0,0),3);



	imshow( "leftImg_h", lh );
	waitKey(1);
	imshow( "rightImg_h", rh );
	waitKey(1);
}

void ImageLoader::show_image_m(int flag)
{
//	printf("left_x=%d, right_x=%d, left_y=%d, right_y=%d\n",left_x_pos,right_x_pos,y_pos,y_pos);
	Mat lm=leftImg.clone();
	Mat rm=rightImg.clone();

	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;
	Point r_a4, r_b4, l_a4, l_b4;

	if (flag)
	{
		r_a4.x=left_x_pos-get_disp2()+82;
		r_a4.y=y_pos+82;
		r_b4.x=left_x_pos-get_disp2()+window_size-83;
		r_b4.y=y_pos+window_size-83;

		rectangle(lm,l_a4,l_b4,(255,255,255),3);
		rectangle(rm,r_a4,r_b4,(255,255,255),3);
	}


	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(lm,l_a1,l_b1,(255,0,0),3);
	rectangle(rm,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(lm,l_a2,l_b2,(255,0,0),3);
	rectangle(rm,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(lm,l_a3,l_b3,(255,0,0),3);
	rectangle(rm,r_a3,r_b3,(255,0,0),3);


	imshow( "leftImg_m", lm );
	waitKey(1);
	imshow( "rightImg_m", rm );
	waitKey(1);
}

void ImageLoader::show_image_h(int flag, int option)
{
//	printf("left_x=%d, right_x=%d, left_y=%d, right_y=%d\n",left_x_pos,right_x_pos,y_pos,y_pos);

	Mat lh=leftImg.clone();
	Mat rh=rightImg.clone();

	Mat lh3(480,640,CV_8U(3));
	cvtColor(lh,lh3,CV_GRAY2RGB);
	Mat rh3(480,640,CV_8U(3));
	cvtColor(rh,rh3,CV_GRAY2RGB);


	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;
	Point r_a4, r_b4, l_a4, l_b4;

	if (flag)
	{
		r_a4.x=left_x_pos-get_disp2()+82;
		r_a4.y=y_pos+82;
		r_b4.x=left_x_pos-get_disp2()+window_size-83;
		r_b4.y=y_pos+window_size-83;

		rectangle(lh3,l_a4,l_b4,Scalar(255,255,255),3);
		rectangle(rh3,r_a4,r_b4,Scalar(255,255,255),3);
	}



	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(lh3,l_a1,l_b1,Scalar(0,0,0),3);
	rectangle(rh3,r_a1,r_b1,Scalar(0,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(lh3,l_a2,l_b2,Scalar(0,0,0),3);
	rectangle(rh3,r_a2,r_b2,Scalar(0,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(lh3,l_a3,l_b3,Scalar(0,0,0),3);
	rectangle(rh3,r_a3,r_b3,Scalar(0,0,0),3);

	switch(option)
	{
	case 0:
		l_a1.x=left_x_pos;
		l_a1.y=y_pos;
		l_b1.x=left_x_pos+window_size;
		l_b1.y=y_pos+window_size;
		r_a1.x=right_x_pos;
		r_a1.y=y_pos;
		r_b1.x=right_x_pos+window_size;
		r_b1.y=y_pos+window_size;

		rectangle(lh3,l_a1,l_b1,Scalar(0,255,0),3);
		rectangle(rh3,r_a1,r_b1,Scalar(0,255,0),3);
		break;

	case 1:
		l_a2.x=left_x_pos+55;
		l_a2.y=y_pos+55;
		l_b2.x=left_x_pos+window_size-55;
		l_b2.y=y_pos+window_size-55;
		r_a2.x=right_x_pos+55;
		r_a2.y=y_pos+55;
		r_b2.x=right_x_pos+window_size-55;
		r_b2.y=y_pos+window_size-55;

		rectangle(lh3,l_a2,l_b2,Scalar(0,255,0),3);
		rectangle(rh3,r_a2,r_b2,Scalar(0,255,0),3);
		break;

	case 2:
		l_a3.x=left_x_pos+82;
		l_a3.y=y_pos+82;
		l_b3.x=left_x_pos+window_size-83;
		l_b3.y=y_pos+window_size-83;
		r_a3.x=right_x_pos+82;
		r_a3.y=y_pos+82;
		r_b3.x=right_x_pos+window_size-83;
		r_b3.y=y_pos+window_size-83;

		rectangle(lh3,l_a3,l_b3,Scalar(0,255,0),3);
		rectangle(rh3,r_a3,r_b3,Scalar(0,255,0),3);
		break;
	}

	imshow( "leftImg_h", lh3 );
	waitKey(1);
	imshow( "rightImg_h", rh3 );
	waitKey(1);
}


void ImageLoader::show_image(Mat left, Mat right)
{
//	printf("left_x=%d, right_x=%d, left_y=%d, right_y=%d\n",left_x_pos,right_x_pos,left_y_pos,right_y_pos);

	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;

	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(left,l_a1,l_b1,(255,0,0),3);
	rectangle(right,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(left,l_a2,l_b2,(255,0,0),3);
	rectangle(right,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(left,l_a3,l_b3,(255,0,0),3);
	rectangle(right,r_a3,r_b3,(255,0,0),3);


	imshow( "leftImg", left );
	waitKey(1);
	imshow( "rightImg", right );
	waitKey(1);
}


void ImageLoader::show_image(Mat left)
{
	Point r_a1, r_b1, l_a1, l_b1;
	Point r_a2, r_b2, l_a2, l_b2;
	Point r_a3, r_b3, l_a3, l_b3;

	l_a1.x=left_x_pos;
	l_a1.y=y_pos;
	l_b1.x=left_x_pos+window_size;
	l_b1.y=y_pos+window_size;
	r_a1.x=right_x_pos;
	r_a1.y=y_pos;
	r_b1.x=right_x_pos+window_size;
	r_b1.y=y_pos+window_size;

	rectangle(left,l_a1,l_b1,(255,0,0),3);
	rectangle(left,r_a1,r_b1,(255,0,0),3);


	l_a2.x=left_x_pos+55;
	l_a2.y=y_pos+55;
	l_b2.x=left_x_pos+window_size-55;
	l_b2.y=y_pos+window_size-55;
	r_a2.x=right_x_pos+55;
	r_a2.y=y_pos+55;
	r_b2.x=right_x_pos+window_size-55;
	r_b2.y=y_pos+window_size-55;

	rectangle(left,l_a2,l_b2,(255,0,0),3);
	rectangle(left,r_a2,r_b2,(255,0,0),3);



	l_a3.x=left_x_pos+82;
	l_a3.y=y_pos+82;
	l_b3.x=left_x_pos+window_size-83;
	l_b3.y=y_pos+window_size-83;
	r_a3.x=right_x_pos+82;
	r_a3.y=y_pos+82;
	r_b3.x=right_x_pos+window_size-83;
	r_b3.y=y_pos+window_size-83;

	rectangle(left,l_a3,l_b3,(255,0,0),3);
	rectangle(left,r_a3,r_b3,(255,0,0),3);




	imshow( "leftImg", left );

	waitKey(1);
}

void ImageLoader::add_jitter()
{
	int version_move=rand()%3-1;
	left_x_pos+=version_move;
	right_x_pos+=version_move;

	int tilt_move=rand()%3-1;
	y_pos+=tilt_move;
	
	if (y_pos<0||y_pos>(image_height-window_size))
		y_pos=(image_height-window_size)*float(rand())/float(RAND_MAX);

}

void ImageLoader::get_fixation_point(int index)
{
	int _coordinate=rand()%(CSMAP_WIDTH*CSMAP_HEIGHT)+index*CSMAP_WIDTH*CSMAP_HEIGHT;

	float randSelection = float(rand())/RAND_MAX;

	for(int i = 0; i < CSMAP_WIDTH*CSMAP_HEIGHT; i++)
	{
		if(randSelection < csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i])
		{
			_coordinate = i;
			break;
		}
		else
			randSelection -= csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i];
	}

	
	y_pos=_coordinate%CSMAP_HEIGHT*2+112-110;
	left_x_pos=_coordinate/CSMAP_HEIGHT*2+152-110;


	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
	

}



void ImageLoader::get_fixation_point(int index, float randPercent)
{
//	printf("randPercent=%f/n",randPercent);
	int _coordinate=rand()%(CSMAP_WIDTH*CSMAP_HEIGHT)+index*CSMAP_WIDTH*CSMAP_HEIGHT;

	float randSelection = randPercent;

	for(int i = 0; i < CSMAP_WIDTH*CSMAP_HEIGHT; i++)
	{
		if(randSelection < csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i])
		{
			_coordinate = i;
			break;
		}
		else
			randSelection -= csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i];
	}

/*
	float max_percent=0;
	int max_index=0;
	
	for(int i = 0; i < CSMAP_WIDTH*CSMAP_HEIGHT; i++)
	{
		if (max_percent<csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i])
		{
			max_percent=csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i];
			max_index=i;
		}
	}
	int _coordinate=max_index;
*/
	y_pos=_coordinate%CSMAP_HEIGHT*2+112-110;
	left_x_pos=_coordinate/CSMAP_HEIGHT*2+192-110;

//	printf("(%d, %d)  (%d, %d)\n",_coordinate/CSMAP_HEIGHT,_coordinate%CSMAP_HEIGHT,left_x_pos,y_pos);

	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
	

}

void ImageLoader::get_fixation_point2(int index, float randPercent)
{
//	printf("randPercent=%f/n",randPercent);
	int _coordinate=rand()%(CSMAP_WIDTH*CSMAP_HEIGHT)+index*CSMAP_WIDTH*CSMAP_HEIGHT;

	float randSelection = randPercent;

	for(int i = 0; i < CSMAP_WIDTH*CSMAP_HEIGHT; i++)
	{
		if(randSelection < csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i])
		{
			_coordinate = i;
			break;
		}
		else
			randSelection -= csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i];
	}

/*
	float max_percent=0;
	int max_index=0;
	
	for(int i = 0; i < CSMAP_WIDTH*CSMAP_HEIGHT; i++)
	{
		if (max_percent<csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i])
		{
			max_percent=csmapCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+i];
			max_index=i;
		}
	}
	int _coordinate=max_index;
*/
	y_pos=_coordinate%CSMAP_HEIGHT*2+112-110;
	left_x_pos=_coordinate/CSMAP_HEIGHT*2+192-110;

//	printf("(%d, %d)  (%d, %d)\n",_coordinate/CSMAP_HEIGHT,_coordinate%CSMAP_HEIGHT,left_x_pos,y_pos);

	dispDiff=localMaxCell[index*CSMAP_WIDTH*CSMAP_HEIGHT+_coordinate];

	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
	

}

float ImageLoader::get_dispMap_xy(int y, int x)
{
	return dispMap[x*IMAGE_HEIGHT+y];
}

float ImageLoader::get_disp()
{
	int sum=0;
	int num=0;
	for (int y=(y_pos+106); y<(y_pos+114); y++)
	{
		for (int x=(left_x_pos+106); x<(left_x_pos+114); x++)
		{
			sum+=dispMap[x*IMAGE_HEIGHT+y];
			num++;
		}
	}	
//	printf("num=%d\n",num);
	return sum/num;
}

float ImageLoader::get_disp2()
{
	int sum=0;
	int num=0;
	for (int y=(y_pos+110-4); y<(y_pos+110+4); y++)
	{
		for (int x=(left_x_pos+110-4); x<(left_x_pos+110+4); x++)
		{
			sum+=dispMapPool[IMAGE_WIDTH*IMAGE_HEIGHT*image_index+x*IMAGE_HEIGHT+y];
			num++;
		}
	}	
//	printf("num=%d\n",num);
	return sum/num;
}


void ImageLoader::load_image()
{
	imageData.clear();
	for(int j = 0; j < TESTNUM; j++)
	{
		char path_Full[200];
		sprintf(path_Full,"D:\\zzt\\Mphil\\Code\\img_dataset\\test5\\imgF%02d.png",j+1);
		imageData.push_back(Mat_<float>(imread(path_Full,CV_LOAD_IMAGE_GRAYSCALE))/255);
	}

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\test5\\csmapCell.mat');");
	tomat::get(csmapCell,"csmapCell",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\test5\\disMapCell.mat');");
	tomat::get(dispMapPool,"disMapCell",0,0);
}



void ImageLoader::get_frame_number_saliency(int i, int interval)
{
	if(i%1000000 == 0)
	{
		
		curEnv = (i/1000000)%2;
		imageData.clear();
		for(int j = 0; j < 180; j++)
		{
			char path_Full[200];
			sprintf(path_Full,"D:\\zzt\\Mphil\\Code\\img_dataset\\trainingSet\\imgF%05d.png",j*5+curEnv*5*180+1);
		
			imageData.push_back(Mat_<float>(imread(path_Full,CV_LOAD_IMAGE_GRAYSCALE))/255);
		}

		char path_Full[200];
		engEvalString(ep,"clear all;");
		sprintf(path_Full,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\trainingSet\\aa_csmap_cell%d.mat');",curEnv+1);
		engEvalString(ep,path_Full);

		tomat::get(csmapCell,"csmap_cell",0,0);
	}

	if (i%(10*interval)==0)
	{
		curFrame=rand()%180;

		char path_Full[200];
		engEvalString(ep,"clear all;");
		sprintf(path_Full,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet\\disMap%05d.mat');",curFrame*5+curEnv*5*180+1);
		engEvalString(ep,path_Full);
		tomat::get(dispMap,"disMap",0,0);
	}

	if (i%interval==0)
	{
		get_fixation_point(curFrame);
	
		right_x_pos=left_x_pos+vergence_angle;

		image_index=curFrame;

	}
	add_jitter();
}

void ImageLoader::get_frame_number_together(int frameIndex, int lx, int ly, int loadFlag)
{
	if (loadFlag==0)
	{
		imageData.clear();
		char path_Full[200];
		for(int j = 0; j < TESTNUM; j++)
		{
			engEvalString(ep,"clear all;");
			sprintf(path_Full,"D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\testF%02d.png",j+1);
			imageData.push_back(Mat_<float>(imread(path_Full,CV_LOAD_IMAGE_GRAYSCALE))/255);

		}
		engEvalString(ep,"clear all;");
		engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\disMapCell.mat');");
		tomat::get(dispMapPool,"disMapCell",0,0);

	}

	image_index=frameIndex;
	left_x_pos=lx;
	y_pos=ly;
	right_x_pos=left_x_pos+vergence_angle;
}

void ImageLoader::get_frame_number_together(int frameIndex, float randPercent, int loadFlag)
{
	if (loadFlag==0)
	{
		imageData.clear();
		char path_Full[200];
		for(int j = 0; j < TESTNUM; j++)
		{
			engEvalString(ep,"clear all;");
			sprintf(path_Full,"D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\testF%02d.png",j+1);
			imageData.push_back(Mat_<float>(imread(path_Full,CV_LOAD_IMAGE_GRAYSCALE))/255);

		}
		engEvalString(ep,"clear all;");
		engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\disMapCell.mat');");
		tomat::get(dispMapPool,"disMapCell",0,0);
		
		engEvalString(ep,"clear all;");
		engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\csmapCell.mat');");
		tomat::get(csmapCell,"csmapCell",0,0);

	}

	image_index=frameIndex;
	get_fixation_point(image_index, randPercent);
	right_x_pos=left_x_pos+vergence_angle;
}

void ImageLoader::get_frame_number_together2(int frameIndex, float randPercent, int loadFlag)
{
	if (loadFlag==0)
	{
		imageData.clear();
		char path_Full[200];
		for(int j = 0; j < TESTNUM; j++)
		{
			engEvalString(ep,"clear all;");
			sprintf(path_Full,"D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\testF%02d.png",j+1);
			imageData.push_back(Mat_<float>(imread(path_Full,CV_LOAD_IMAGE_GRAYSCALE))/255);

		}
		engEvalString(ep,"clear all;");
		engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\disMapCell.mat');");
		tomat::get(dispMapPool,"disMapCell",0,0);
		
		engEvalString(ep,"clear all;");
		engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\newCsmapCell.mat');");
		tomat::get(csmapCell,"newCsmapCell",0,0);

		engEvalString(ep,"clear all;");
		engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet2\\localMaxCell.mat');");
		tomat::get(localMaxCell,"localMaxCell",0,0);

	}

	image_index=frameIndex;
	get_fixation_point2(image_index, randPercent);
	right_x_pos=left_x_pos+vergence_angle;
}



void ImageLoader::get_image_input1(int frameIndex, int lx, int ly, int action_taken)
{
	left_x_pos=lx;
	y_pos=ly;
	image_index=frameIndex;

	vergence_angle-=action_taken;


	imageData[image_index].colRange(0,image_width).convertTo(leftImg,CV_32FC1);
	imageData[image_index].colRange(image_width+1,2*image_width).convertTo(rightImg,CV_32FC1);

	right_x_pos=left_x_pos+vergence_angle;

	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
		

	if (right_x_pos<0 || right_x_pos>=(image_width-window_size))
	{
		right_x_pos=rand()%(image_width-window_size);
		vergence_angle=right_x_pos-left_x_pos;
	}

	leftImg.rowRange(y_pos,y_pos+window_size)
		   .colRange(left_x_pos,left_x_pos+window_size)
		   .convertTo(leftWtmp,CV_32FC1);

	rightImg.rowRange(y_pos,y_pos+window_size)
		    .colRange(right_x_pos,right_x_pos+window_size)
		    .convertTo(rightWtmp,CV_32FC1);
	

	
	leftWin=leftWtmp;
	rightWin=rightWtmp;


//	show_image(leftImg,rightImg);
	
}

void ImageLoader::get_image_input1(int frameIndex, int lx, int ly, int action_taken, int disp, int flag)
{
	left_x_pos=lx;
	y_pos=ly;
	image_index=frameIndex;

	if (flag)
	{
		char path_Full[200];
		engEvalString(ep,"clear all;");
		sprintf(path_Full,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet\\disMap%05d.mat');",image_index*10+1);
		engEvalString(ep,path_Full);
		tomat::get(dispMap,"disMap",0,0);
	}

	if (flag)
		vergence_angle=-get_disp()+disp;
	else
		vergence_angle-=action_taken;


	imageData[image_index].colRange(0,image_width).convertTo(leftImg,CV_32FC1);
	imageData[image_index].colRange(image_width+1,2*image_width).convertTo(rightImg,CV_32FC1);

	right_x_pos=left_x_pos+vergence_angle;

	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
		

	if (right_x_pos<0 || right_x_pos>=(image_width-window_size))
	{
		right_x_pos=rand()%(image_width-window_size);
		vergence_angle=right_x_pos-left_x_pos;
	}

	leftImg.rowRange(y_pos,y_pos+window_size)
		   .colRange(left_x_pos,left_x_pos+window_size)
		   .convertTo(leftWtmp,CV_32FC1);

	rightImg.rowRange(y_pos,y_pos+window_size)
		    .colRange(right_x_pos,right_x_pos+window_size)
		    .convertTo(rightWtmp,CV_32FC1);
	

	
	leftWin=leftWtmp;
	rightWin=rightWtmp;


//	show_image(leftImg,rightImg);
	
}

void ImageLoader::get_fixation_point_saliency(int frameIndex, int flag, float randPercent)
{
	image_index=frameIndex;

	if (flag)
	{
		char path_Full[200];
		engEvalString(ep,"clear all;");
		sprintf(path_Full,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\test5\\new_smap%02d.mat');",frameIndex+1);
		engEvalString(ep,path_Full);
		tomat::get(new_csmap,"new_smap",0,0);
	}

	int _coordinate=rand()%(CSMAP_WIDTH*CSMAP_HEIGHT);

	float randSelection = randPercent;

	for(int i = 0; i < CSMAP_WIDTH*CSMAP_HEIGHT; i++)
	{
		if(randSelection < csmapCell[i])
		{
			_coordinate = i;
			break;
		}
		else
			randSelection -= csmapCell[i];
	}

	
	y_pos=_coordinate%CSMAP_HEIGHT*2+112-110;
	left_x_pos=_coordinate/CSMAP_HEIGHT*2+152-110;


	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
	
	
}

void ImageLoader::get_fixation_point_max(int frameIndex, int flag)
{
	image_index=frameIndex;

	if (flag)
	{
		char path_Full[200];
		engEvalString(ep,"clear all;");
		sprintf(path_Full,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\test5\\new_smap%02d.mat');",frameIndex+1);
		engEvalString(ep,path_Full);
		tomat::get(new_csmap,"new_smap",0,0);
	}

	float max_value=0;
	int	max_index=0;
	for (int ii=0; ii<CSMAP_WIDTH*CSMAP_HEIGHT; ii++)
	{
		

		if (max_value<new_csmap[ii])
		{
			max_value=new_csmap[ii];
			max_index=ii;
		}
	}

	y_pos=max_index%CSMAP_HEIGHT*2+112-110;
	left_x_pos=max_index/CSMAP_HEIGHT*2+192-110;
	
	int cy=max_index%CSMAP_HEIGHT;
	int cx=max_index/CSMAP_HEIGHT;

	for (int y=0; y<FILT_RANGE; y++)
	{
		for (int x=0; x<FILT_RANGE; x++)
		{
			int range=(FILT_RANGE-1)/2;
			if((cy+(y-range)>=0)&&(cy+(y-range)<CSMAP_HEIGHT)&&(cx+(x-range)>=0)&&(cx+(x-range)<CSMAP_WIDTH))
				new_csmap[(cx+x-range)*CSMAP_HEIGHT+cy+y-range]=new_csmap[(cx+x-range)*CSMAP_HEIGHT+cy+y-range]*discount_filter[x*FILT_RANGE+y];
		}
	}
}

void ImageLoader::get_fixation_point_max2(int frameIndex, int reload)
{
	image_index=frameIndex;

	if (reload)
	{
		char path_Full[200];
		engEvalString(ep,"clear all;");
		engEvalString(ep,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\test5\\csmapCell.mat');");
		tomat::get(csmapCell,"csmapCell",0,0);
	}

	float max_value=0;
	int	max_index=0;
	for (int ii=0; ii<CSMAP_WIDTH*CSMAP_HEIGHT; ii++)
	{
		
		if (max_value<csmapCell[image_index*CSMAP_WIDTH*CSMAP_HEIGHT+ii])
		{
			max_value=csmapCell[image_index*CSMAP_WIDTH*CSMAP_HEIGHT+ii];
			max_index=ii;
		}
	}

	y_pos=max_index%CSMAP_HEIGHT*2+112-110;
	left_x_pos=max_index/CSMAP_HEIGHT*2+192-110;
	
	int cy=max_index%CSMAP_HEIGHT;
	int cx=max_index/CSMAP_HEIGHT;

	for (int y=0; y<FILT_RANGE; y++)
	{
		for (int x=0; x<FILT_RANGE; x++)
		{
			int range=(FILT_RANGE-1)/2;
			if((cy+(y-range)>=0)&&(cy+(y-range)<CSMAP_HEIGHT)&&(cx+(x-range)>=0)&&(cx+(x-range)<CSMAP_WIDTH))
				csmapCell[image_index*CSMAP_WIDTH*CSMAP_HEIGHT+(cx+x-range)*CSMAP_HEIGHT+cy+y-range]=new_csmap[image_index*CSMAP_WIDTH*CSMAP_HEIGHT+(cx+x-range)*CSMAP_HEIGHT+cy+y-range]*discount_filter[x*range+y];
		}
	}
}


void ImageLoader::get_image_input2(int frameIndex, int action_taken, int disp, int flag)
{
	imageData.clear();

	char path_Full[200];
	sprintf(path_Full,"D:\\zzt\\Mphil\\Code\\img_dataset\\trainingSet\\imgF%05d.png",frameIndex*5+1);
	imageData.push_back(Mat_<float>(imread(path_Full,CV_LOAD_IMAGE_GRAYSCALE))/255);
	

	image_index=0;

	if (flag)
	{
		char path_Full[200];
		engEvalString(ep,"clear all;");
		sprintf(path_Full,"load('D:\\zzt\\Mphil\\Code\\img_dataset\\testingSet\\disMap%05d.mat');",frameIndex*5+1);
		engEvalString(ep,path_Full);
		tomat::get(dispMap,"disMap",0,0);
	}

	if (flag)
		vergence_angle=-get_disp()+disp;
	else
		vergence_angle-=action_taken;


	imageData[image_index].colRange(0,image_width).convertTo(leftImg,CV_32FC1);
	imageData[image_index].colRange(image_width+1,2*image_width).convertTo(rightImg,CV_32FC1);

	right_x_pos=left_x_pos+vergence_angle;

	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
		

	if (right_x_pos<0 || right_x_pos>=(image_width-window_size))
	{
		right_x_pos=rand()%(image_width-window_size);
		vergence_angle=right_x_pos-left_x_pos;
	}

	leftImg.rowRange(y_pos,y_pos+window_size)
		   .colRange(left_x_pos,left_x_pos+window_size)
		   .convertTo(leftWtmp,CV_32FC1);

	rightImg.rowRange(y_pos,y_pos+window_size)
		    .colRange(right_x_pos,right_x_pos+window_size)
		    .convertTo(rightWtmp,CV_32FC1);
	

	
	leftWin=leftWtmp;
	rightWin=rightWtmp;


//	show_image(leftImg,rightImg);
	
}



void ImageLoader::window_position_after_action_stereo(int action)
{
	vergence_angle-=action;

	if (vergence_angle<-200 || vergence_angle>50)
		vergence_angle=-rand()%100;

	imageData[image_index].colRange(0,image_width).convertTo(leftImg,CV_32FC1);
	imageData[image_index].colRange(image_width+1,2*image_width).convertTo(rightImg,CV_32FC1);

	right_x_pos=left_x_pos+vergence_angle;

	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
		

	if (right_x_pos<0 || right_x_pos>=(image_width-window_size))
	{
		right_x_pos=rand()%(image_width-window_size);
		vergence_angle=right_x_pos-left_x_pos;
	}

	leftImg.rowRange(y_pos,y_pos+window_size)
		   .colRange(left_x_pos,left_x_pos+window_size)
		   .convertTo(leftWtmp,CV_32FC1);

	rightImg.rowRange(y_pos,y_pos+window_size)
		    .colRange(right_x_pos,right_x_pos+window_size)
		    .convertTo(rightWtmp,CV_32FC1);
	

	
	leftWin=leftWtmp;
	rightWin=rightWtmp;


//	show_image(leftImg,rightImg);
	
}



void ImageLoader::window_position_after_action_stereo(int fovea_disparity, int i, int action, int interval)
{
	if (i%interval==0)
		vergence_angle=-get_disp2()+fovea_disparity;
	else
		vergence_angle-=action;

	imageData[image_index].colRange(0,image_width).convertTo(leftImg,CV_32FC1);
	imageData[image_index].colRange(image_width+1,2*image_width).convertTo(rightImg,CV_32FC1);

	right_x_pos=left_x_pos+vergence_angle;

	if ((left_x_pos>=image_width-window_size) || (left_x_pos<0))
		left_x_pos=rand()%(image_width-window_size);

	if ((y_pos>=image_height-window_size) || (y_pos<0))
		y_pos=rand()%(image_height-window_size);
		

	if (right_x_pos<0 || right_x_pos>=(image_width-window_size))
	{
		right_x_pos=rand()%(image_width-window_size);
		vergence_angle=right_x_pos-left_x_pos;
	}

	leftImg.rowRange(y_pos,y_pos+window_size)
		   .colRange(left_x_pos,left_x_pos+window_size)
		   .convertTo(leftWtmp,CV_32FC1);

	rightImg.rowRange(y_pos,y_pos+window_size)
		    .colRange(right_x_pos,right_x_pos+window_size)
		    .convertTo(rightWtmp,CV_32FC1);
	

	
	leftWin=leftWtmp;
	rightWin=rightWtmp;


//	show_image(leftImg,rightImg);
	
}



void ImageLoader::window_position_after_action_mono(int fovea_disparity)
{
	right_x_pos=left_x_pos+fovea_disparity;

	imageData[image_index].colRange(0,image_width).convertTo(leftImg,CV_32FC1);


	leftImg.rowRange(y_pos,y_pos+window_size)
		   .colRange(left_x_pos,left_x_pos+window_size)
		   .convertTo(leftWtmp,CV_32FC1);

	leftImg.rowRange(y_pos,y_pos+window_size)
		    .colRange(right_x_pos,right_x_pos+window_size)
		    .convertTo(rightWtmp,CV_32FC1);
	

	
	leftWin=leftWtmp;
	rightWin=rightWtmp;

	
}



void ImageLoader::window_position_after_action_mono(int fovea_disparity, int i, int action, int interval)
{
	if (i%interval==0)
		right_x_pos=left_x_pos+fovea_disparity;

	else
		right_x_pos-=action;


	if (right_x_pos<0 || right_x_pos>=(image_width-window_size))
	{
		right_x_pos=rand()%(image_width-window_size);
		vergence_angle=right_x_pos-left_x_pos;
	}

	imageData[image_index].colRange(0,image_width).convertTo(leftImg,CV_32FC1);


	leftImg.rowRange(y_pos,y_pos+window_size)
		   .colRange(left_x_pos,left_x_pos+window_size)
		   .convertTo(leftWtmp,CV_32FC1);

	leftImg.rowRange(y_pos,y_pos+window_size)
		    .colRange(right_x_pos,right_x_pos+window_size)
		    .convertTo(rightWtmp,CV_32FC1);
	

	
	leftWin=leftWtmp;
	rightWin=rightWtmp;

//	show_image(leftImg);

}