#include"RealVergence.h"


using namespace std;
SYSTEMTIME sst0;
SYSTEMTIME sst1;


/******************* Declaration Region ************************/

int initCuda();
void destroyCuda();
void trainJoint();
void testPolicyStdAlone();
void testPolicyStdAlone1();

void performance_error_mono();
void performance_error_bino();
void performance_error_mono1();
void performance_error_bino1();


void performance_error_bino5();
void performance_error_bino51();


void performance_error_bino_compare();
void performance_error_bino_compare1();
void performance_error_bino_compare2();

void performance_error_bino_compare_realTime();

void performance_error_bino_compare_max();

void performance_error_bino_compare_max_result();

void performance_error_bino_compare_max_result2();

void performance_error_bino_compare_max_result3();
/******************* Main Function ******************************/

int main()
{
	initCuda();
	
//	performance_error_bino_compare_realTime();


//	performance_error_bino_compare_max();

//	performance_error_bino_compare_max_result();

	performance_error_bino_compare_max_result2();

//	performance_error_bino_compare_max_result3();

	destroyCuda();


	
	system("pause");
	return 0;
}


/******************** Vergence Functions *************************/

void trainJoint()
{
	cudaError_t  stat;

	float* filter;

	

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	
	tomat::start();
	engEvalString(ep,"cd('D:\\zzt\\Mphil\\Code\\new_dataset_saliency_together_elegant_whitening_debug\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);
	


	int option_action=0;
	int action_index=(ACTIONNUM-1)/2;

	ImageLoader Image;
	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());




	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);


	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);


	clock_t startTime = clock();
	clock_t checkTime = clock();
	cudaEvent_t start, stop;
	stat=cudaEventCreate(&start);
	stat=cudaEventCreate(&stop);
	stat=cudaEventRecord(start,0);
	float timeInterval=0;
	int image_interval = 10;
	int check_interval = 1000;
 
	double t0,t1,t2;

	float option_record[SCALENUM]={0.0};
	float J_vector[SCALENUM+1]={0.0};

	int record_index=0;

	float* debug_leftWh;

	for(int i = 0; i < T_TRAIN; i++)
	{
		if(i == drecord.checkPoints[drecord.currCp])
		{
			drecord.saveBases44(asmodel44.bases1,asmodel44.bases2);
			drecord.saveBases24(asmodel24.bases1,asmodel24.bases2);
			drecord.saveBases22(asmodel22.bases1,asmodel22.bases2);
			drecord.saveBases13(asmodel13.bases1,asmodel13.bases2);
			drecord.saveBases12(asmodel12.bases1,asmodel12.bases2);
			drecord.saveBases11(asmodel11.bases1,asmodel11.bases2);


			drecord.saveWeights4(rlmodel4.getPWeights(), rlmodel4.getVWeights(), rlmodel4.getNWeights());
			drecord.saveWeights2(rlmodel2.getPWeights(), rlmodel2.getVWeights(), rlmodel2.getNWeights());
			drecord.saveWeights1(rlmodel1.getPWeights(), rlmodel1.getVWeights(), rlmodel1.getNWeights());
			drecord.saveWeights0(option.getPWeights(), option.getVWeights(), option.getNWeights());


			drecord.currCp++;
			
			tomat::push(FoveaBatch44.getLfovea(),FILT_WIDTH,FILT_WIDTH,"fovea44L",1,0);
			tomat::push(FoveaBatch24.getLfovea(),FILT_WIDTH,FILT_WIDTH,"fovea24L",1,0);
			tomat::push(FoveaBatch22.getLfovea(),FILT_WIDTH,FILT_WIDTH,"fovea22L",1,0);
			tomat::push(FoveaBatch13.getLfovea(),FILT_WIDTH,FILT_WIDTH,"fovea13L",1,0);
			tomat::push(FoveaBatch12.getLfovea(),FILT_WIDTH,FILT_WIDTH,"fovea12L",1,0);
			tomat::push(FoveaBatch11.getLfovea(),FILT_WIDTH,FILT_WIDTH,"fovea11L",1,0);

			tomat::push(FoveaBatch44.getLfoveaWh(),FoveaBatch44.getFoveaWidth(),FoveaBatch44.getFoveaWidth(),"fovea44LWh",1,0);
			tomat::push(FoveaBatch24.getLfoveaWh(),FoveaBatch24.getFoveaWidth(),FoveaBatch24.getFoveaWidth(),"fovea24LWh",1,0);
			tomat::push(FoveaBatch22.getLfoveaWh(),FoveaBatch22.getFoveaWidth(),FoveaBatch22.getFoveaWidth(),"fovea22LWh",1,0);
			tomat::push(FoveaBatch13.getLfoveaWh(),FoveaBatch13.getFoveaWidth(),FoveaBatch13.getFoveaWidth(),"fovea13LWh",1,0);
			tomat::push(FoveaBatch12.getLfoveaWh(),FoveaBatch12.getFoveaWidth(),FoveaBatch12.getFoveaWidth(),"fovea12LWh",1,0);
			tomat::push(FoveaBatch11.getLfoveaWh(),FoveaBatch11.getFoveaWidth(),FoveaBatch11.getFoveaWidth(),"fovea11LWh",1,0);
			

			tomat::push(FoveaBatch44.getBatch(),AGENTSNUM,BASISDIM,"batch44",1,0);
			tomat::push(FoveaBatch24.getBatch(),AGENTSNUM,BASISDIM,"batch24",1,0);
			tomat::push(FoveaBatch22.getBatch(),AGENTSNUM,BASISDIM,"batch22",1,0);
			tomat::push(FoveaBatch13.getBatch(),AGENTSNUM,BASISDIM,"batch13",1,0);
			tomat::push(FoveaBatch12.getBatch(),AGENTSNUM,BASISDIM,"batch12",1,0);
			tomat::push(FoveaBatch11.getBatch(),AGENTSNUM,BASISDIM,"batch11",1,0);
			
			tomat::push(asmodel44.bases1,BASISDIM,BASESNUM,"b44a",1,0);
			tomat::push(asmodel44.bases2,BASISDIM,BASESNUM,"b44b",1,0);
			tomat::push(asmodel24.bases1,BASISDIM,BASESNUM,"b24a",1,0);
			tomat::push(asmodel24.bases2,BASISDIM,BASESNUM,"b24b",1,0);
			tomat::push(asmodel22.bases1,BASISDIM,BASESNUM,"b22a",1,0);
			tomat::push(asmodel22.bases2,BASISDIM,BASESNUM,"b22b",1,0);
			tomat::push(asmodel13.bases1,BASISDIM,BASESNUM,"b13a",1,0);
			tomat::push(asmodel13.bases2,BASISDIM,BASESNUM,"b13b",1,0);
			tomat::push(asmodel12.bases1,BASISDIM,BASESNUM,"b12a",1,0);
			tomat::push(asmodel12.bases2,BASISDIM,BASESNUM,"b12b",1,0);
			tomat::push(asmodel11.bases1,BASISDIM,BASESNUM,"b11a",1,0);
			tomat::push(asmodel11.bases2,BASISDIM,BASESNUM,"b11b",1,0);


			tomat::push(rlmodel4.getPWeights(),ACTIONNUM,FEATUREDIM,"polw4",1,0);
			tomat::push(rlmodel2.getPWeights(),ACTIONNUM,2*FEATUREDIM,"polw2",1,0);
			tomat::push(rlmodel1.getPWeights(),ACTIONNUM,3*FEATUREDIM,"polw1",1,0);
			tomat::push(option.getPWeights(),3,6*FEATUREDIM,"polw0",1,0);

			tomat::push(rlmodel4.getVWeights(),1,FEATUREDIM,"valw4",1,0);
			tomat::push(rlmodel2.getVWeights(),1,2*FEATUREDIM,"valw2",1,0);
			tomat::push(rlmodel1.getVWeights(),1,3*FEATUREDIM,"valw1",1,0);
		    tomat::push(option.getVWeights(),1,FEATUREDIM*6,"valw0",1,0);


			tomat::push(rlmodel4.getNWeights(),ACTIONNUM,FEATUREDIM,"ww4",1,0);
			tomat::push(rlmodel2.getNWeights(),ACTIONNUM,2*FEATUREDIM,"ww2",1,0);
			tomat::push(rlmodel1.getNWeights(),ACTIONNUM,3*FEATUREDIM,"ww1",1,0);
			tomat::push(option.getNWeights(),3,6*FEATUREDIM,"ww0",1,0);

			J_vector[0]=option.getAverageReward();
			J_vector[1]=rlmodel4.getAverageReward();
			J_vector[2]=rlmodel2.getAverageReward();
			J_vector[3]=rlmodel1.getAverageReward();

			tomat::push(J_vector,1,4,"J_vector",0,0);
			


			record_index++;
			char des_name[200];
			sprintf(des_name,"inner_record%d.mat",record_index);

			char tmpCmd[100];
			sprintf(tmpCmd,"save('%s')",des_name);
			engEvalString(ep,tmpCmd);
		}
	
		Image.get_frame_number_saliency(i,10);
	
		Image.window_position_after_action_stereo(vergence_command[action_index]);
		cudaThreadSynchronize();


		FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();		
		FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();	
		FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();		
		FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();		
		FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();

		asmodel44.AssomEncode();
		cudaThreadSynchronize();
		asmodel24.AssomEncode();
		cudaThreadSynchronize();
		asmodel22.AssomEncode();
		cudaThreadSynchronize();
		asmodel13.AssomEncode();
		cudaThreadSynchronize();
		asmodel12.AssomEncode();
		cudaThreadSynchronize();
		asmodel11.AssomEncode();
		cudaThreadSynchronize();

		option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
		cudaThreadSynchronize();

		rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
		cudaThreadSynchronize();
		rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
		cudaThreadSynchronize();
		rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
		cudaThreadSynchronize();



		rlmodel4.updateNetwork(i, action_index);
		cudaThreadSynchronize();
		rlmodel2.updateNetwork(i, action_index);
		cudaThreadSynchronize();
		rlmodel1.updateNetwork(i, action_index);
		cudaThreadSynchronize();
	
		option.updateNetwork(i);
		cudaThreadSynchronize();

		option_action=option.rlGetAction();

		option_record[option_action]+=1;


		switch (option_action+1)
		{
		case 1:
			action_index=rlmodel4.rlGetAction();
			break;
		
		case 2:		
			action_index=rlmodel2.rlGetAction();
			break;

		case 3:		
			action_index=rlmodel1.rlGetAction();
			break;
		
		default:
			printf("bug\n");
			system("pause");
		}


		cudaThreadSynchronize();
		asmodel44.updateBases(); 
		cudaThreadSynchronize();
		asmodel24.updateBases(); 
		cudaThreadSynchronize();
		asmodel22.updateBases(); 
		cudaThreadSynchronize();
		asmodel13.updateBases(); 
		cudaThreadSynchronize();
		asmodel12.updateBases(); 
		cudaThreadSynchronize();
		asmodel11.updateBases(); 
		cudaThreadSynchronize();
			
		if(!(i%check_interval))
		{
			cudaThreadSynchronize();
			stat=cudaEventRecord(stop);
			stat=cudaEventSynchronize(stop);
			stat=cudaEventElapsedTime(&timeInterval,start,stop);
			stat=cudaEventRecord(start);
			cudaEventSynchronize(start);

			int timeLeft = int(timeInterval/1000 * float(T_TRAIN-i)/float(check_interval+EPS));
			//int timeLeft = int(timeInterval/1000*T_TRAIN);
			int h = timeLeft/3600;
			int m = (timeLeft%3600)/60;
			int s = timeLeft%60;
			printf("Percentage: %5.1f%%; \t",i*100.0/T_TRAIN);
			if(h > 0)
				printf("Time left: %d hours, %d minutes, %d seconds\n", h,m,s);
			else if(m > 0)
				printf("Time left: %d minutes, %d seconds\n", m,s);
			else
				printf("Time left: %d seconds\n",s);


		}

	} //end of training

	//test policy

	drecord.saveToMat();
	//testPolicyStdAlone();
	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_4_24_1.mat");
	engEvalString(ep,tmpCmd);


	testPolicyStdAlone();
	testPolicyStdAlone1();
}





int initCuda()
{
	seed=13;//51R 52R 21L  1Bal 
	srand (seed); //13, 10, //23
	cublasStatus_t stat;
	 
	cublasOne = 1;
	cublasZero = 0;
	cublasNegOne = -1;
	
	
	stat=cublasCreate(&cuhandle);
	

	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return -1;
	}

	for(int i = 0; i < BLASSTREAMS; i ++)
	{
		cudaStreamCreate(&streams[i]);
		cublasCreate(&cuhandles[i]);
		cublasSetStream(cuhandles[i],streams[i]);
	}

	drecord.create(T_TRAIN);
	return 0;


}

void destroyCuda()
{
	cublasDestroy(cuhandle);
	for(int i = 0; i < BLASSTREAMS; i ++)
	{
		cublasDestroy(cuhandles[i]);
		cudaStreamDestroy(streams[i]);
	}

}

void testPolicyStdAlone()
{
	
	const int disparity_range=20;
	const int testSize=100;
	float policy4[(2*disparity_range+1)*ACTIONNUM] ={0.0} ;
	float policy2[(2*disparity_range+1)*ACTIONNUM] ={0.0} ;
	float policy1[(2*disparity_range+1)*ACTIONNUM] ={0.0} ;
	float policy0[(2*disparity_range+1)*SCALENUM] ={0.0} ;


	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
	



	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_whitening_debug\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('drecord_4_24_1.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);


	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	for(int act=0; act<(2*disparity_range+1); act++)
	{
		int curDis = act-disparity_range;	
		printf("curDis=%d \n",curDis);

		for(int t=0; t<testSize; t++)
		{
			int i=act*testSize+t;
			Image.get_frame_number_saliency(i,1);
	
			Image.window_position_after_action_mono(curDis);
			cudaThreadSynchronize();

			FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
	    	cudaThreadSynchronize();
		    FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		    cudaThreadSynchronize();
		    FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		    cudaThreadSynchronize();
		    FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		    cudaThreadSynchronize();
		    FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		    cudaThreadSynchronize();
		    FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		    cudaThreadSynchronize();


			asmodel44.AssomEncode();
			cudaThreadSynchronize();
			asmodel24.AssomEncode();
			cudaThreadSynchronize();
			asmodel22.AssomEncode();
			cudaThreadSynchronize();
			asmodel13.AssomEncode();
			cudaThreadSynchronize();
			asmodel12.AssomEncode();
			cudaThreadSynchronize();
			asmodel11.AssomEncode();
			cudaThreadSynchronize();

			option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
			cudaThreadSynchronize();

			rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
			cudaThreadSynchronize();
			rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
			cudaThreadSynchronize();
			rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
			cudaThreadSynchronize();

			rlmodel4.softmaxAct(0);
			cudaThreadSynchronize();
			rlmodel2.softmaxAct(0);	
			cudaThreadSynchronize();
			rlmodel1.softmaxAct(0);	
			cudaThreadSynchronize();
		    option.softmaxAct(0);
			cudaThreadSynchronize();

			for(int k=0; k<ACTIONNUM; k++)
			{
				policy4[k+ACTIONNUM*act] += rlmodel4.hostPolicy[k];
				policy2[k+ACTIONNUM*act] += rlmodel2.hostPolicy[k];
				policy1[k+ACTIONNUM*act] += rlmodel1.hostPolicy[k];
			}

			for(int k=0; k<SCALENUM; k++)
				policy0[k+SCALENUM*act] += option.hostPolicy[k];

			}

		}

	tomat::push(policy4,ACTIONNUM,(2*disparity_range+1),"totalPolicy4",0,0);
	tomat::push(policy2,ACTIONNUM,(2*disparity_range+1),"totalPolicy2",0,0);
	tomat::push(policy1,ACTIONNUM,(2*disparity_range+1),"totalPolicy1",0,0);
	tomat::push(policy0,SCALENUM,(2*disparity_range+1),"totalPolicy0",0,0);

	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_4_24_2.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);
}

void testPolicyStdAlone1()
{
	
	const int disparity_range=20;
	const int testSize=100;
	const int checkNum=CHECKPOINTSNUM;
	float policy4[(2*disparity_range+1)*ACTIONNUM*checkNum] ={0.0} ;
	float policy2[(2*disparity_range+1)*ACTIONNUM*checkNum] ={0.0} ;
	float policy1[(2*disparity_range+1)*ACTIONNUM*checkNum] ={0.0} ;
	float policy0[(2*disparity_range+1)*SCALENUM*checkNum] ={0.0} ;


	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
	



	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3*checkNum);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*checkNum);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*checkNum);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*checkNum);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*checkNum);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_whitening_debug\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('drecord_4_24_1.mat');");

	tomat::get(valWeights4,"valW4",0,0);
	tomat::get(valWeights2,"valW2",0,0);
	tomat::get(valWeights1,"valW1",0,0);
	tomat::get(valWeights0,"valW0",0,0);

	tomat::get(polWeights4,"polW4",0,0);
	tomat::get(polWeights2,"polW2",0,0);
	tomat::get(polWeights1,"polW1",0,0);
	tomat::get(polWeights0,"polW0",0,0);

	tomat::get(bases44a,"B44a",0,0);
	tomat::get(bases44b,"B44b",0,0);

	tomat::get(bases24a,"B24a",0,0);
	tomat::get(bases24b,"B24b",0,0);

	tomat::get(bases22a,"B22a",0,0);
	tomat::get(bases22b,"B22b",0,0);

	tomat::get(bases13a,"B13a",0,0);
	tomat::get(bases13b,"B13b",0,0);

	tomat::get(bases12a,"B12a",0,0);
	tomat::get(bases12b,"B12b",0,0);

	tomat::get(bases11a,"B11a",0,0);
	tomat::get(bases11b,"B11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);



	for (int checkPoint=0; checkPoint<checkNum; checkPoint+=2)
	{
		asmodel44.setBases(&bases44a[checkPoint*BASISDIM*BASESNUM],&bases44b[checkPoint*BASISDIM*BASESNUM]);
		asmodel24.setBases(&bases24a[checkPoint*BASISDIM*BASESNUM],&bases24b[checkPoint*BASISDIM*BASESNUM]);
		asmodel22.setBases(&bases22a[checkPoint*BASISDIM*BASESNUM],&bases22b[checkPoint*BASISDIM*BASESNUM]);
		asmodel13.setBases(&bases13a[checkPoint*BASISDIM*BASESNUM],&bases13b[checkPoint*BASISDIM*BASESNUM]);
		asmodel12.setBases(&bases12a[checkPoint*BASISDIM*BASESNUM],&bases12b[checkPoint*BASISDIM*BASESNUM]);
		asmodel11.setBases(&bases11a[checkPoint*BASISDIM*BASESNUM],&bases11b[checkPoint*BASISDIM*BASESNUM]);

		rlmodel4.setWeights(&polWeights4[checkPoint*FEATUREDIM*ACTIONNUM],&valWeights4[checkPoint*FEATUREDIM]);
		rlmodel2.setWeights(&polWeights2[checkPoint*2*FEATUREDIM*ACTIONNUM],&valWeights2[checkPoint*2*FEATUREDIM]);
		rlmodel1.setWeights(&polWeights1[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights1[checkPoint*3*FEATUREDIM]);
	
		option.setWeights(&polWeights0[checkPoint*FEATUREDIM*6*SCALENUM],&valWeights0[checkPoint*FEATUREDIM*6]);
		printf("\n\ncheckPoint=%d\n",checkPoint);

		for(int act=0; act<(2*disparity_range+1); act++)
		{
			int curDis = act-disparity_range;	
			printf("%d ",curDis);

			for(int t=0; t<testSize; t++)
			{
				int i=act*testSize+t;
				Image.get_frame_number_saliency(i,1);
	
				Image.window_position_after_action_mono(curDis);
				cudaThreadSynchronize();

				FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
	    		cudaThreadSynchronize();
				FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
				cudaThreadSynchronize();
				FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
				cudaThreadSynchronize();
				FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
				cudaThreadSynchronize();
				FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
				cudaThreadSynchronize();
				FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
				cudaThreadSynchronize();


				asmodel44.AssomEncode();
				cudaThreadSynchronize();
				asmodel24.AssomEncode();
				cudaThreadSynchronize();
				asmodel22.AssomEncode();
				cudaThreadSynchronize();
				asmodel13.AssomEncode();
				cudaThreadSynchronize();
				asmodel12.AssomEncode();
				cudaThreadSynchronize();
				asmodel11.AssomEncode();
				cudaThreadSynchronize();

				option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
											 ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
				cudaThreadSynchronize();
				rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
				cudaThreadSynchronize();
				rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.softmaxAct(0);
				cudaThreadSynchronize();
				rlmodel2.softmaxAct(0);	
				cudaThreadSynchronize();
				rlmodel1.softmaxAct(0);	
				cudaThreadSynchronize();
				option.softmaxAct(0);
				cudaThreadSynchronize();

				for(int k=0; k<11; k++)
				{
					int index=checkPoint*(2*disparity_range+1)*11+k+11*act;
					policy4[index] += rlmodel4.hostPolicy[k];
					policy2[index] += rlmodel2.hostPolicy[k];
					policy1[index] += rlmodel1.hostPolicy[k];
				}

				for(int k=0; k<3; k++)
					policy0[checkPoint*(2*disparity_range+1)*3+k+3*act] += option.hostPolicy[k];

			}
		}
	}
	tomat::push(policy4,ACTIONNUM*(2*disparity_range+1),checkNum,"totalPolicy4",0,0);
	tomat::push(policy2,ACTIONNUM*(2*disparity_range+1),checkNum,"totalPolicy2",0,0);
	tomat::push(policy1,ACTIONNUM*(2*disparity_range+1),checkNum,"totalPolicy1",0,0);
	tomat::push(policy0,SCALENUM*(2*disparity_range+1),checkNum,"totalPolicy0",0,0);

	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_4_24_3.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);
}

void performance_error_mono()
{
	const int iterNum=5;
	const int disparity_range=20;
	const int testSize=100;
	int action_record[iterNum] ={0.0} ;
	float error_record[2*disparity_range+1]={0.0};
	const int tolerance=3;
	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};

	
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);

	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('drecord_4_24_1.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);


	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	int option_index=0;
	int action_taken=0;
	int action_equal=0;

	for(int act=0; act<(2*disparity_range+1); act++)
	{
		int curDis = act-disparity_range;	
		printf("curDis=%d \n",curDis);
		
		for(int t=0; t<testSize; t++)
		{
			int i=act*testSize+t;
			Image.get_frame_number_saliency(i,1);

			for (int iteration=0; iteration<iterNum; iteration++)
			{
				Image.window_position_after_action_mono(curDis, iteration, action_taken,iterNum);

				FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   	cudaThreadSynchronize();
			    FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();


				asmodel44.AssomEncode();
				cudaThreadSynchronize();
				asmodel24.AssomEncode();
				cudaThreadSynchronize();
				asmodel22.AssomEncode();
				cudaThreadSynchronize();
				asmodel13.AssomEncode();
				cudaThreadSynchronize();
				asmodel12.AssomEncode();
				cudaThreadSynchronize();
				asmodel11.AssomEncode();
				cudaThreadSynchronize();

				option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
				cudaThreadSynchronize();
				rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
				cudaThreadSynchronize();
				rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.softmaxAct(0);
				cudaThreadSynchronize();
				rlmodel2.softmaxAct(0);	
				cudaThreadSynchronize();
				rlmodel1.softmaxAct(0);	
				cudaThreadSynchronize();
			    option.softmaxAct(0);
				cudaThreadSynchronize();

				rlmodel4.greedyAction();
				cudaThreadSynchronize();
				rlmodel2.greedyAction();
				cudaThreadSynchronize();
				rlmodel1.greedyAction();
				cudaThreadSynchronize();
				option.greedyAction();
				cudaThreadSynchronize();


				option_index=option.rlGetAction();

	//			printf("option_index=%d\n",option_index);
	//			system("pause");

				switch (option_index+1)
				{
				case 1:
					action_taken=vergence_command[rlmodel4.rlGetAction()];
					break;
		
				case 2:		
					action_taken=vergence_command[rlmodel2.rlGetAction()];
					break;

				case 3:		
					action_taken=vergence_command[rlmodel1.rlGetAction()];
					break;


				default:
					printf("bug\n");
					system("pause");
				}
									
				action_record[iteration]=action_taken;
			}	

			action_equal=0;
			for (int ii=0;ii<iterNum;ii++)
				action_equal+=action_record[ii];

			//	printf("action_equal=%d\n\n",action_equal);

			if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
				action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
			actionEq_record[action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
		}

	}

	tomat::push(actionEq_record,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record",0,0);



	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_4_24_12.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);

}

void performance_error_bino()
{
	const int iterNum=5;
	const int disparity_range=20;
	const int testSize=100;
	int action_record[iterNum] ={0.0} ;
	float error_record[2*disparity_range+1]={0.0};
	const int tolerance=3;
	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};

	
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);

	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('drecord_4_26_10.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);


	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	int option_index=0;
	int action_taken=0;
	int action_equal=0;

	for(int act=0; act<(2*disparity_range+1); act++)
	{
		int curDis = act-disparity_range;	
		printf("curDis=%d \n",curDis);
		
		for(int t=0; t<testSize; t++)
		{
			int i=act*testSize+t;
			Image.get_frame_number_saliency(i,1);

			for (int iteration=0; iteration<iterNum; iteration++)
			{
				Image.window_position_after_action_stereo(curDis, iteration, action_taken, iterNum);

				FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   	cudaThreadSynchronize();
			    FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();


				asmodel44.AssomEncode();
				cudaThreadSynchronize();
				asmodel24.AssomEncode();
				cudaThreadSynchronize();
				asmodel22.AssomEncode();
				cudaThreadSynchronize();
				asmodel13.AssomEncode();
				cudaThreadSynchronize();
				asmodel12.AssomEncode();
				cudaThreadSynchronize();
				asmodel11.AssomEncode();
				cudaThreadSynchronize();

				option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
				cudaThreadSynchronize();
				rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
				cudaThreadSynchronize();
				rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.softmaxAct(0);
				cudaThreadSynchronize();
				rlmodel2.softmaxAct(0);	
				cudaThreadSynchronize();
				rlmodel1.softmaxAct(0);	
				cudaThreadSynchronize();
			    option.softmaxAct(0);
				cudaThreadSynchronize();

				rlmodel4.greedyAction();
				cudaThreadSynchronize();
				rlmodel2.greedyAction();
				cudaThreadSynchronize();
				rlmodel1.greedyAction();
				cudaThreadSynchronize();
				option.greedyAction();
				cudaThreadSynchronize();


				option_index=option.rlGetAction();

				printf("option_index=%d\n",option_index);
				system("pause");

				switch (option_index+1)
				{
				case 1:
					action_taken=vergence_command[rlmodel4.rlGetAction()];
					break;
		
				case 2:		
					action_taken=vergence_command[rlmodel2.rlGetAction()];
					break;

				case 3:		
					action_taken=vergence_command[rlmodel1.rlGetAction()];
					break;


				default:
					printf("bug\n");
					system("pause");
				}
									
				action_record[iteration]=action_taken;
			}	

			action_equal=0;
			for (int ii=0;ii<iterNum;ii++)
				action_equal+=action_record[ii];

			//	printf("action_equal=%d\n\n",action_equal);

			if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
				action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
			actionEq_record[action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
		}

	}

	tomat::push(actionEq_record,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record",0,0);



	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_4_26_113.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);

}

void performance_error_mono1()
{
	const int iterNum=5;
	const int checkNum=30;
	const int disparity_range=20;
	const int testSize=100;
	int action_record[iterNum] ={0.0} ;
	float error_record[2*disparity_range+1]={0.0};
	const int tolerance=3;
	float actionEq_record[(2*disparity_range+1)*(2*(disparity_range+tolerance)+1)*checkNum]={0.0};

	
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);

	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3*checkNum);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*checkNum);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*checkNum);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*checkNum);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*checkNum);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('drecord_4_24_1.mat');");

	tomat::get(valWeights4,"valW4",0,0);
	tomat::get(valWeights2,"valW2",0,0);
	tomat::get(valWeights1,"valW1",0,0);
	tomat::get(valWeights0,"valW0",0,0);

	tomat::get(polWeights4,"polW4",0,0);
	tomat::get(polWeights2,"polW2",0,0);
	tomat::get(polWeights1,"polW1",0,0);
	tomat::get(polWeights0,"polW0",0,0);

	tomat::get(bases44a,"B44a",0,0);
	tomat::get(bases44b,"B44b",0,0);

	tomat::get(bases24a,"B24a",0,0);
	tomat::get(bases24b,"B24b",0,0);

	tomat::get(bases22a,"B22a",0,0);
	tomat::get(bases22b,"B22b",0,0);

	tomat::get(bases13a,"B13a",0,0);
	tomat::get(bases13b,"B13b",0,0);

	tomat::get(bases12a,"B12a",0,0);
	tomat::get(bases12b,"B12b",0,0);

	tomat::get(bases11a,"B11a",0,0);
	tomat::get(bases11b,"B11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);


	int option_index=0;
	int action_taken=0;
	int action_equal=0;



	for (int checkPoint=0; checkPoint<checkNum; checkPoint++)
	{
		asmodel44.setBases(&bases44a[checkPoint*BASISDIM*BASESNUM],&bases44b[checkPoint*BASISDIM*BASESNUM]);
		asmodel24.setBases(&bases24a[checkPoint*BASISDIM*BASESNUM],&bases24b[checkPoint*BASISDIM*BASESNUM]);
		asmodel22.setBases(&bases22a[checkPoint*BASISDIM*BASESNUM],&bases22b[checkPoint*BASISDIM*BASESNUM]);
		asmodel13.setBases(&bases13a[checkPoint*BASISDIM*BASESNUM],&bases13b[checkPoint*BASISDIM*BASESNUM]);
		asmodel12.setBases(&bases12a[checkPoint*BASISDIM*BASESNUM],&bases12b[checkPoint*BASISDIM*BASESNUM]);
		asmodel11.setBases(&bases11a[checkPoint*BASISDIM*BASESNUM],&bases11b[checkPoint*BASISDIM*BASESNUM]);

		rlmodel4.setWeights(&polWeights4[checkPoint*FEATUREDIM*ACTIONNUM],&valWeights4[checkPoint*FEATUREDIM]);
		rlmodel2.setWeights(&polWeights2[checkPoint*2*FEATUREDIM*ACTIONNUM],&valWeights2[checkPoint*2*FEATUREDIM]);
		rlmodel1.setWeights(&polWeights1[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights1[checkPoint*3*FEATUREDIM]);
	
		option.setWeights(&polWeights0[checkPoint*FEATUREDIM*6*SCALENUM],&valWeights0[checkPoint*FEATUREDIM*6]);
		printf("\n\ncheckPoint=%d\n",checkPoint);

		for(int act=0; act<(2*disparity_range+1); act++)
		{
			int curDis = act-disparity_range;	
			printf("%d ",curDis);
		
			for(int t=0; t<testSize; t++)
			{
				int i=act*testSize+t;
				Image.get_frame_number_saliency(i,1);

				for (int iteration=0; iteration<iterNum; iteration++)
				{
					Image.window_position_after_action_mono(curDis, iteration, action_taken,iterNum);

					FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   		cudaThreadSynchronize();
					FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();


					asmodel44.AssomEncode();
					cudaThreadSynchronize();
					asmodel24.AssomEncode();
					cudaThreadSynchronize();
					asmodel22.AssomEncode();
					cudaThreadSynchronize();
					asmodel13.AssomEncode();
					cudaThreadSynchronize();
					asmodel12.AssomEncode();
					cudaThreadSynchronize();
					asmodel11.AssomEncode();
					cudaThreadSynchronize();

					option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
											 ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					cudaThreadSynchronize();

					rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
					cudaThreadSynchronize();
					rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
					cudaThreadSynchronize();
					rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					cudaThreadSynchronize();

					rlmodel4.softmaxAct(0);
					cudaThreadSynchronize();
					rlmodel2.softmaxAct(0);	
					cudaThreadSynchronize();
					rlmodel1.softmaxAct(0);	
					cudaThreadSynchronize();
					option.softmaxAct(0);
					cudaThreadSynchronize();


					rlmodel4.greedyAction();
					cudaThreadSynchronize();
					rlmodel2.greedyAction();
					cudaThreadSynchronize();
					rlmodel1.greedyAction();
					cudaThreadSynchronize();
					option.greedyAction();
					cudaThreadSynchronize();


					option_index=option.rlGetAction();

					switch (option_index+1)
					{
					case 1:
						action_taken=vergence_command[rlmodel4.rlGetAction()];
						break;
		
					case 2:		
						action_taken=vergence_command[rlmodel2.rlGetAction()];
						break;

					case 3:		
						action_taken=vergence_command[rlmodel1.rlGetAction()];
						break;


					default:
						printf("bug\n");
						system("pause");
					}
									
					action_record[iteration]=action_taken;
				}	

				action_equal=0;
				for (int ii=0;ii<iterNum;ii++)
					action_equal+=action_record[ii];

				//	printf("action_equal=%d\n\n",action_equal);

				if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
					action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
				
				int _index_=action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act+(2*(disparity_range+tolerance)+1)*(2*disparity_range+1)*checkPoint;
				actionEq_record[_index_]++;
			}

		}
	}

	tomat::push(actionEq_record,(2*(disparity_range+tolerance)+1)*(2*disparity_range+1),checkNum,"actionEq_record",0,0);



	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_4_24_22.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);

}

void performance_error_bino1()
{
	const int iterNum=5;
	const int checkNum=30;
	const int disparity_range=20;
	const int testSize=100;
	int action_record[iterNum] ={0.0} ;
	float error_record[2*disparity_range+1]={0.0};
	const int tolerance=3;
	float actionEq_record[(2*disparity_range+1)*(2*(disparity_range+tolerance)+1)*checkNum]={0.0};

	
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);

	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3*checkNum);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*checkNum);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*checkNum);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*checkNum);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*checkNum);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('debug_4_24_1.mat');");

	tomat::get(valWeights4,"valW4",0,0);
	tomat::get(valWeights2,"valW2",0,0);
	tomat::get(valWeights1,"valW1",0,0);
	tomat::get(valWeights0,"valW0",0,0);

	tomat::get(polWeights4,"polW4",0,0);
	tomat::get(polWeights2,"polW2",0,0);
	tomat::get(polWeights1,"polW1",0,0);
	tomat::get(polWeights0,"polW0",0,0);

	tomat::get(bases44a,"B44a",0,0);
	tomat::get(bases44b,"B44b",0,0);

	tomat::get(bases24a,"B24a",0,0);
	tomat::get(bases24b,"B24b",0,0);

	tomat::get(bases22a,"B22a",0,0);
	tomat::get(bases22b,"B22b",0,0);

	tomat::get(bases13a,"B13a",0,0);
	tomat::get(bases13b,"B13b",0,0);

	tomat::get(bases12a,"B12a",0,0);
	tomat::get(bases12b,"B12b",0,0);

	tomat::get(bases11a,"B11a",0,0);
	tomat::get(bases11b,"B11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);


	int option_index=0;
	int action_taken=0;
	int action_equal=0;


	for (int checkPoint=0; checkPoint<checkNum; checkPoint++)
	{
		asmodel44.setBases(&bases44a[checkPoint*BASISDIM*BASESNUM],&bases44b[checkPoint*BASISDIM*BASESNUM]);
		asmodel24.setBases(&bases24a[checkPoint*BASISDIM*BASESNUM],&bases24b[checkPoint*BASISDIM*BASESNUM]);
		asmodel22.setBases(&bases22a[checkPoint*BASISDIM*BASESNUM],&bases22b[checkPoint*BASISDIM*BASESNUM]);
		asmodel13.setBases(&bases13a[checkPoint*BASISDIM*BASESNUM],&bases13b[checkPoint*BASISDIM*BASESNUM]);
		asmodel12.setBases(&bases12a[checkPoint*BASISDIM*BASESNUM],&bases12b[checkPoint*BASISDIM*BASESNUM]);
		asmodel11.setBases(&bases11a[checkPoint*BASISDIM*BASESNUM],&bases11b[checkPoint*BASISDIM*BASESNUM]);

		rlmodel4.setWeights(&polWeights4[checkPoint*FEATUREDIM*ACTIONNUM],&valWeights4[checkPoint*FEATUREDIM]);
		rlmodel2.setWeights(&polWeights2[checkPoint*2*FEATUREDIM*ACTIONNUM],&valWeights2[checkPoint*2*FEATUREDIM]);
		rlmodel1.setWeights(&polWeights1[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights1[checkPoint*3*FEATUREDIM]);
	
		option.setWeights(&polWeights0[checkPoint*FEATUREDIM*6*SCALENUM],&valWeights0[checkPoint*FEATUREDIM*6]);
		printf("\n\ncheckPoint=%d\n",checkPoint);

		for(int act=0; act<(2*disparity_range+1); act++)
		{
			int curDis = act-disparity_range;	
			printf("%d ",curDis);
		
			for(int t=0; t<testSize; t++)
			{
				int i=act*testSize+t;
				Image.get_frame_number_saliency(i,1);

				for (int iteration=0; iteration<iterNum; iteration++)
				{
					Image.window_position_after_action_stereo(curDis, iteration, action_taken, 5);

					FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   		cudaThreadSynchronize();
					FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();


					asmodel44.AssomEncode();
					cudaThreadSynchronize();
					asmodel24.AssomEncode();
					cudaThreadSynchronize();
					asmodel22.AssomEncode();
					cudaThreadSynchronize();
					asmodel13.AssomEncode();
					cudaThreadSynchronize();
					asmodel12.AssomEncode();
					cudaThreadSynchronize();
					asmodel11.AssomEncode();
					cudaThreadSynchronize();

					option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
											 ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					cudaThreadSynchronize();

					rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
					cudaThreadSynchronize();
					rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
					cudaThreadSynchronize();
					rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					cudaThreadSynchronize();

					rlmodel4.softmaxAct(0);
					cudaThreadSynchronize();
					rlmodel2.softmaxAct(0);	
					cudaThreadSynchronize();
					rlmodel1.softmaxAct(0);	
					cudaThreadSynchronize();
					option.softmaxAct(0);
					cudaThreadSynchronize();


					rlmodel4.greedyAction();
					cudaThreadSynchronize();
					rlmodel2.greedyAction();
					cudaThreadSynchronize();
					rlmodel1.greedyAction();
					cudaThreadSynchronize();
					option.greedyAction();
					cudaThreadSynchronize();


					option_index=option.rlGetAction();

					switch (option_index+1)
					{
					case 1:
						action_taken=vergence_command[rlmodel4.rlGetAction()];
						break;
		
					case 2:		
						action_taken=vergence_command[rlmodel2.rlGetAction()];
						break;

					case 3:		
						action_taken=vergence_command[rlmodel1.rlGetAction()];
						break;


					default:
						printf("bug\n");
						system("pause");
					}
									
					action_record[iteration]=action_taken;
				}	

				action_equal=0;
				for (int ii=0;ii<iterNum;ii++)
					action_equal+=action_record[ii];

				//	printf("action_equal=%d\n\n",action_equal);

				if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
					action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
				
				int _index_=action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act+(2*(disparity_range+tolerance)+1)*(2*disparity_range+1)*checkPoint;
				actionEq_record[_index_]++;
			}
		}
	}
	tomat::push(actionEq_record,(2*(disparity_range+tolerance)+1)*(2*disparity_range+1),checkNum,"actionEq_record",0,0);



	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_4_24_23.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);

}

void performance_error_bino5()
{
	
	const int disparity_range=20;
	const int testSize=100;
	int action_record[10] ={0.0} ;
	float error_record[2*disparity_range+1]={0.0};
	const int tolerance=3;
	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};

	
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);

	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_whitening_debug\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('drecord_4_24_1.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);


	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	int option_index=0;
	int action_taken=0;
	int action_equal[5]={0};


	for(int act=0; act<(2*disparity_range+1); act++)
	{
		int curDis = act-disparity_range;	
		printf("curDis=%d \n",curDis);
		
		for(int t=0; t<testSize; t++)
		{
			int i=act*testSize+t;
			Image.get_frame_number_saliency(i,1);

			for (int iteration=0; iteration<10; iteration++)
			{
				Image.window_position_after_action_stereo(curDis, iteration, action_taken,10);

				FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   	cudaThreadSynchronize();
			    FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();


				asmodel44.AssomEncode();
				cudaThreadSynchronize();
				asmodel24.AssomEncode();
				cudaThreadSynchronize();
				asmodel22.AssomEncode();
				cudaThreadSynchronize();
				asmodel13.AssomEncode();
				cudaThreadSynchronize();
				asmodel12.AssomEncode();
				cudaThreadSynchronize();
				asmodel11.AssomEncode();
				cudaThreadSynchronize();

				option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
				cudaThreadSynchronize();
				rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
				cudaThreadSynchronize();
				rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				cudaThreadSynchronize();

				rlmodel4.softmaxAct(0);
				cudaThreadSynchronize();
				rlmodel2.softmaxAct(0);	
				cudaThreadSynchronize();
				rlmodel1.softmaxAct(0);	
				cudaThreadSynchronize();
			    option.softmaxAct(0);
				cudaThreadSynchronize();


				option_index=option.rlGetAction();

				switch (option_index+1)
				{
				case 1:
					action_taken=vergence_command[rlmodel4.rlGetAction()];
					break;
		
				case 2:		
					action_taken=vergence_command[rlmodel2.rlGetAction()];
					break;

				case 3:		
					action_taken=vergence_command[rlmodel1.rlGetAction()];
					break;


				default:
					printf("bug\n");
					system("pause");
				}
									
				action_record[iteration]=action_taken;
			}	

			for (int jj=0; jj<5; jj++)
			{
				for (int ii=0;ii<5+jj+1;ii++)
					action_equal[jj]+=action_record[ii];
			
			//	printf("action_equal=%d\n\n",action_equal);

				if ((action_equal[jj]>disparity_range+tolerance) || (action_equal[jj]<-disparity_range-tolerance))
					action_equal[jj]=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);

				actionEq_record[action_equal[jj]+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
			}
		}

	}

	tomat::push(actionEq_record,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record",0,0);



	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_3_24_12.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);

}

void performance_error_bino51()
{
	const int checkNum=20;
	const int disparity_range=20;
	const int testSize=100;
	int action_record[10] ={0.0} ;
	float error_record[2*disparity_range+1]={0.0};
	const int tolerance=3;
	float actionEq_record[(2*disparity_range+1)*(2*(disparity_range+tolerance)+1)*checkNum]={0.0};

	
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);

	printf("Start testing policy....\n");

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3*checkNum);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*checkNum);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*checkNum);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*checkNum);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*checkNum);

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_whitening_debug\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('drecord_4_24_1.mat');");

	tomat::get(valWeights4,"valW4",0,0);
	tomat::get(valWeights2,"valW2",0,0);
	tomat::get(valWeights1,"valW1",0,0);
	tomat::get(valWeights0,"valW0",0,0);

	tomat::get(polWeights4,"polW4",0,0);
	tomat::get(polWeights2,"polW2",0,0);
	tomat::get(polWeights1,"polW1",0,0);
	tomat::get(polWeights0,"polW0",0,0);

	tomat::get(bases44a,"B44a",0,0);
	tomat::get(bases44b,"B44b",0,0);

	tomat::get(bases24a,"B24a",0,0);
	tomat::get(bases24b,"B24b",0,0);

	tomat::get(bases22a,"B22a",0,0);
	tomat::get(bases22b,"B22b",0,0);

	tomat::get(bases13a,"B13a",0,0);
	tomat::get(bases13b,"B13b",0,0);

	tomat::get(bases12a,"B12a",0,0);
	tomat::get(bases12b,"B12b",0,0);

	tomat::get(bases11a,"B11a",0,0);
	tomat::get(bases11b,"B11b",0,0);

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);


	ImageLoader Image;

	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());


	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);


	int option_index=0;
	int action_taken=0;
	int action_equal[5]={0};


	for (int checkPoint=0; checkPoint<checkNum; checkPoint++)
	{
		asmodel44.setBases(&bases44a[checkPoint*BASISDIM*BASESNUM],&bases44b[checkPoint*BASISDIM*BASESNUM]);
		asmodel24.setBases(&bases24a[checkPoint*BASISDIM*BASESNUM],&bases24b[checkPoint*BASISDIM*BASESNUM]);
		asmodel22.setBases(&bases22a[checkPoint*BASISDIM*BASESNUM],&bases22b[checkPoint*BASISDIM*BASESNUM]);
		asmodel13.setBases(&bases13a[checkPoint*BASISDIM*BASESNUM],&bases13b[checkPoint*BASISDIM*BASESNUM]);
		asmodel12.setBases(&bases12a[checkPoint*BASISDIM*BASESNUM],&bases12b[checkPoint*BASISDIM*BASESNUM]);
		asmodel11.setBases(&bases11a[checkPoint*BASISDIM*BASESNUM],&bases11b[checkPoint*BASISDIM*BASESNUM]);

		rlmodel4.setWeights(&polWeights4[checkPoint*FEATUREDIM*ACTIONNUM],&valWeights4[checkPoint*FEATUREDIM]);
		rlmodel2.setWeights(&polWeights2[checkPoint*2*FEATUREDIM*ACTIONNUM],&valWeights2[checkPoint*2*FEATUREDIM]);
		rlmodel1.setWeights(&polWeights1[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights1[checkPoint*3*FEATUREDIM]);
	
		option.setWeights(&polWeights0[checkPoint*FEATUREDIM*6*SCALENUM],&valWeights0[checkPoint*FEATUREDIM*6]);
		printf("\n\ncheckPoint=%d\n",checkPoint);

		for(int act=0; act<(2*disparity_range+1); act++)
		{
			int curDis = act-disparity_range;	
			printf("curDis=%d \n",curDis);
		
			for(int t=0; t<testSize; t++)
			{
				int i=act*testSize+t;
				Image.get_frame_number_saliency(i,1);

				for (int iteration=0; iteration<5; iteration++)
				{
					Image.window_position_after_action_stereo(curDis, iteration, action_taken, 5);

					FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   		cudaThreadSynchronize();
					FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();


					asmodel44.AssomEncode();
					cudaThreadSynchronize();
					asmodel24.AssomEncode();
					cudaThreadSynchronize();
					asmodel22.AssomEncode();
					cudaThreadSynchronize();
					asmodel13.AssomEncode();
					cudaThreadSynchronize();
					asmodel12.AssomEncode();
					cudaThreadSynchronize();
					asmodel11.AssomEncode();
					cudaThreadSynchronize();

					option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
											 ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					cudaThreadSynchronize();

					rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
					cudaThreadSynchronize();
					rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
					cudaThreadSynchronize();
					rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					cudaThreadSynchronize();

					rlmodel4.softmaxAct(0);
					cudaThreadSynchronize();
					rlmodel2.softmaxAct(0);	
					cudaThreadSynchronize();
					rlmodel1.softmaxAct(0);	
					cudaThreadSynchronize();
					option.softmaxAct(0);
					cudaThreadSynchronize();


					option_index=option.rlGetAction();

					switch (option_index+1)
					{
					case 1:
						action_taken=vergence_command[rlmodel4.rlGetAction()];
						break;
		
					case 2:		
						action_taken=vergence_command[rlmodel2.rlGetAction()];
						break;

					case 3:		
						action_taken=vergence_command[rlmodel1.rlGetAction()];
						break;


					default:
						printf("bug\n");
						system("pause");
					}
									
					action_record[iteration]=action_taken;
				}	

				for (int jj=0; jj<5; jj++)
				{
					for (int ii=0;ii<5+jj+1;ii++)
						action_equal[jj]+=action_record[ii];
			
				//	printf("action_equal=%d\n\n",action_equal);

					if ((action_equal[jj]>disparity_range+tolerance) || (action_equal[jj]<-disparity_range-tolerance))
						action_equal[jj]=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);

					int _index_=action_equal[jj]+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act+(2*(disparity_range+tolerance)+1)*(2*disparity_range+1)*checkPoint;
					actionEq_record[_index_]++;
				}
			}
		}
	}
	tomat::push(actionEq_record,(2*(disparity_range+tolerance)+1)*(2*disparity_range+1),checkNum,"actionEq_record",0,0);



	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","drecord_3_24_12.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);

}


void performance_error_bino_compare()
{
	const int iterNum=5;
	const int disparity_range=20;
	const int testSize=200;

	int action_record[iterNum] ={0.0};
	int action_record_m[iterNum] ={0.0};

	float error_record[2*disparity_range+1]={0.0};
	float error_record_m[2*disparity_range+1]={0.0};

	const int tolerance=10;
	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};
	float actionEq_record_m[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};
	

	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error_compare\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('inner_record40.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial3.mat');");

	tomat::get(valWeights,"valw",0,0);
	tomat::get(polWeights,"polw",0,0);

	tomat::get(bases11_m,"b11",0,0);
	tomat::get(bases12_m,"b12",0,0);

	tomat::get(bases21_m,"b21",0,0);
	tomat::get(bases22_m,"b22",0,0);

	tomat::get(bases31_m,"b31",0,0);
	tomat::get(bases32_m,"b32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

/*****************************************************************************************************************/
	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);

	asmodel1_m.setBases(bases11_m,bases12_m);
	asmodel2_m.setBases(bases21_m,bases22_m);
	asmodel3_m.setBases(bases31_m,bases32_m);

	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	rlmodel.setWeights(polWeights,valWeights);


/*****************************************************************************************************************/

	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int lx,ly;
	int frameIndex;


	for(int act=0; act<(2*disparity_range+1); act++)
	{
	//	act=37;
		int curDis =act-disparity_range;	
		printf("curDis=%d \n",curDis);

		for(int t=0; t<testSize; t++)
		{
			int i=act*testSize+t;

//			lx=rand()%(640-220-100)+100;
//			ly=rand()%(480-220);
			if (i%10==0)
				frameIndex=rand()%TESTNUM;
			float randPercent=float(rand())/RAND_MAX;
			Image.get_frame_number_together2(frameIndex,randPercent,i);
			Image_m.get_frame_number_together2(frameIndex,randPercent,i);

			printf("\ndispDiff=%f\n",Image.dispDiff);

			for (int iteration=0; iteration<iterNum; iteration++)
			{
/****************************************************************************************************************************************************/
				Image.window_position_after_action_stereo(curDis, iteration, action_taken, iterNum);
				Image_m.window_position_after_action_stereo(curDis, iteration, action_taken_m, iterNum);
/****************************************************************************************************************************************************/

				Image.show_image_h();
				Image_m.show_image_m();

				if (i==0&&iteration==0)
				{
					printf("waiting......");
					while(1)
					{
						Image.show_image_h();
						Image_m.show_image_m();

						if (kbhit())
						{
							if (getch())
								break;
						}
					}
				}


/****************************************************************************************************************************************************/
				FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   	cudaThreadSynchronize();
			    FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();

				FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();
				FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();
				FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();

/****************************************************************************************************************************************************/

				asmodel44.AssomEncode();
				cudaThreadSynchronize();
				asmodel24.AssomEncode();
				cudaThreadSynchronize();
				asmodel22.AssomEncode();
				cudaThreadSynchronize();
				asmodel13.AssomEncode();
				cudaThreadSynchronize();
				asmodel12.AssomEncode();
				cudaThreadSynchronize();
				asmodel11.AssomEncode();
				cudaThreadSynchronize();

				asmodel1_m.AssomEncode();
				cudaThreadSynchronize();	
				asmodel2_m.AssomEncode();
				cudaThreadSynchronize();
				asmodel3_m.AssomEncode();
				cudaThreadSynchronize();

/****************************************************************************************************************************************************/

				option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
				rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
				rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				
				rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
/****************************************************************************************************************************************************/
				rlmodel4.softmaxAct(0);
				rlmodel2.softmaxAct(0);	
				rlmodel1.softmaxAct(0);	
			    option.softmaxAct(0);
			
				rlmodel.softmaxAct(0);

/****************************************************************************************************************************************************/
				rlmodel4.greedyAction();
				rlmodel2.greedyAction();
				rlmodel1.greedyAction();
			    option.greedyAction();
				
				rlmodel.greedyAction();

/****************************************************************************************************************************************************/
				option_index=option.rlGetAction();

				printf("option_index=%d\n",option_index);
				system("pause");

				switch (option_index+1)
				{
				case 1:
					action_taken=vergence_command[rlmodel4.rlGetAction()];
					break;
		
				case 2:		
					action_taken=vergence_command[rlmodel2.rlGetAction()];
					break;

				case 3:		
					action_taken=vergence_command[rlmodel1.rlGetAction()];
					break;


				default:
					printf("bug\n");
					system("pause");
				}
				
				action_taken_m=vergence_command[rlmodel.rlGetAction()];	


				action_record[iteration]=action_taken;
				action_record_m[iteration]=action_taken_m;
			}	
			
			action_equal=0;
			action_equal_m=0;
			for (int ii=0;ii<iterNum;ii++)
			{
				action_equal+=action_record[ii];
				action_equal_m+=action_record_m[ii];
			}
		
			if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
				action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
			actionEq_record[action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;

			if ((action_equal_m>disparity_range+tolerance) || (action_equal_m<-disparity_range-tolerance))
				action_equal_m=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
			actionEq_record_m[action_equal_m+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
		}

	}

	tomat::push(actionEq_record,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record",0,0);
	tomat::push(actionEq_record_m,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record_m",0,0);


	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","inner_compare_5_3_1.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}

void performance_error_bino_compare1()
{
	const int checkNum=30;
	const int iterNum=5;
	const int disparity_range=20;
	const int testSize=200;

	int action_record[iterNum] ={0.0};
	int action_record_m[iterNum] ={0.0};

	float error_record[2*disparity_range+1]={0.0};
	float error_record_m[2*disparity_range+1]={0.0};

	const int tolerance=10;
	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};
	float actionEq_record_m[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};
	

	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM*checkNum);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*checkNum);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3*checkNum);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*checkNum);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*checkNum);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*checkNum);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*checkNum);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error_compare\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('inner_record40.mat');");

	tomat::get(valWeights4,"valW4",0,0);
	tomat::get(valWeights2,"valW2",0,0);
	tomat::get(valWeights1,"valW1",0,0);
	tomat::get(valWeights0,"valW0",0,0);

	tomat::get(polWeights4,"polW4",0,0);
	tomat::get(polWeights2,"polW2",0,0);
	tomat::get(polWeights1,"polW1",0,0);
	tomat::get(polWeights0,"polW0",0,0);

	tomat::get(bases44a,"B44a",0,0);
	tomat::get(bases44b,"B44b",0,0);

	tomat::get(bases24a,"B24a",0,0);
	tomat::get(bases24b,"B24b",0,0);

	tomat::get(bases22a,"B22a",0,0);
	tomat::get(bases22b,"B22b",0,0);

	tomat::get(bases13a,"B13a",0,0);
	tomat::get(bases13b,"B13b",0,0);

	tomat::get(bases12a,"B12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"B11a",0,0);
	tomat::get(bases11b,"B11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial3.mat');");

	tomat::get(valWeights,"valW",0,0);
	tomat::get(polWeights,"polW",0,0);

	tomat::get(bases11_m,"B11",0,0);
	tomat::get(bases12_m,"B12",0,0);

	tomat::get(bases21_m,"B21",0,0);
	tomat::get(bases22_m,"B22",0,0);

	tomat::get(bases31_m,"B31",0,0);
	tomat::get(bases32_m,"B32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);


/*****************************************************************************************************************/

	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int lx,ly;
	int frameIndex;

	for (int checkPoint=0; checkPoint<checkNum; checkPoint++)
	{
		asmodel44.setBases(&bases44a[checkPoint*BASISDIM*BASESNUM],&bases44b[checkPoint*BASISDIM*BASESNUM]);
		asmodel24.setBases(&bases24a[checkPoint*BASISDIM*BASESNUM],&bases24b[checkPoint*BASISDIM*BASESNUM]);
		asmodel22.setBases(&bases22a[checkPoint*BASISDIM*BASESNUM],&bases22b[checkPoint*BASISDIM*BASESNUM]);
		asmodel13.setBases(&bases13a[checkPoint*BASISDIM*BASESNUM],&bases13b[checkPoint*BASISDIM*BASESNUM]);
		asmodel12.setBases(&bases12a[checkPoint*BASISDIM*BASESNUM],&bases12b[checkPoint*BASISDIM*BASESNUM]);
		asmodel11.setBases(&bases11a[checkPoint*BASISDIM*BASESNUM],&bases11b[checkPoint*BASISDIM*BASESNUM]);

		asmodel1_m.setBases(&bases11_m[checkPoint*BASISDIM*BASESNUM],&bases12_m[checkPoint*BASISDIM*BASESNUM]);
		asmodel2_m.setBases(&bases21_m[checkPoint*BASISDIM*BASESNUM],&bases22_m[checkPoint*BASISDIM*BASESNUM]);
		asmodel3_m.setBases(&bases31_m[checkPoint*BASISDIM*BASESNUM],&bases32_m[checkPoint*BASISDIM*BASESNUM]);

		rlmodel4.setWeights(&polWeights4[checkPoint*FEATUREDIM*ACTIONNUM],&valWeights4[checkPoint*FEATUREDIM]);
		rlmodel2.setWeights(&polWeights2[checkPoint*2*FEATUREDIM*ACTIONNUM],&valWeights2[checkPoint*2*FEATUREDIM]);
		rlmodel1.setWeights(&polWeights1[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights1[checkPoint*3*FEATUREDIM]);
	
		option.setWeights(&polWeights0[checkPoint*FEATUREDIM*6*SCALENUM],&valWeights0[checkPoint*FEATUREDIM*6]);

		rlmodel.setWeights(&polWeights[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights[checkPoint*3*FEATUREDIM*ACTIONNUM]);




		for(int act=0; act<(2*disparity_range+1); act++)
		{

		//	act=37;
			int curDis =act-disparity_range;	
			printf("curDis=%d \n",curDis);

			for(int t=0; t<testSize; t++)
			{
				int i=act*testSize+t;

	//			lx=rand()%(640-220-100)+100;
	//			ly=rand()%(480-220);
				if (i%10==0)
					frameIndex=rand()%TESTNUM;
				float randPercent=float(rand())/RAND_MAX;
				Image.get_frame_number_together(frameIndex,randPercent,i);
				Image_m.get_frame_number_together(frameIndex,randPercent,i);

				for (int iteration=0; iteration<iterNum; iteration++)
				{
	/****************************************************************************************************************************************************/
					Image.window_position_after_action_stereo(curDis, iteration, action_taken, iterNum);
					Image_m.window_position_after_action_stereo(curDis, iteration, action_taken_m, iterNum);
	/****************************************************************************************************************************************************/
	/*
					Image.show_image_h();
					Image_m.show_image_m();

					if (i==0&&iteration==0)
					{
						printf("waiting......");
						while(1)
						{
							Image.show_image_h();
							Image_m.show_image_m();

							if (kbhit())
							{
								if (getch())
									break;
							}
						}
					}
	*/

	/****************************************************************************************************************************************************/
					FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   		cudaThreadSynchronize();
					FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();

					FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();
					FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();
					FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();

	/****************************************************************************************************************************************************/

					asmodel44.AssomEncode();
					cudaThreadSynchronize();
					asmodel24.AssomEncode();
					cudaThreadSynchronize();
					asmodel22.AssomEncode();
					cudaThreadSynchronize();
					asmodel13.AssomEncode();
					cudaThreadSynchronize();
					asmodel12.AssomEncode();
					cudaThreadSynchronize();
					asmodel11.AssomEncode();
					cudaThreadSynchronize();

					asmodel1_m.AssomEncode();
					cudaThreadSynchronize();	
					asmodel2_m.AssomEncode();
					cudaThreadSynchronize();
					asmodel3_m.AssomEncode();
					cudaThreadSynchronize();

	/****************************************************************************************************************************************************/

					option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
											 ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
					rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
					rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				
					rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
	/****************************************************************************************************************************************************/
					rlmodel4.softmaxAct(0);
					rlmodel2.softmaxAct(0);	
					rlmodel1.softmaxAct(0);	
					option.softmaxAct(0);
			
					rlmodel.softmaxAct(0);

	/****************************************************************************************************************************************************/
					rlmodel4.greedyAction();
					rlmodel2.greedyAction();
					rlmodel1.greedyAction();
					option.greedyAction();
				
					rlmodel.greedyAction();

	/****************************************************************************************************************************************************/
					option_index=option.rlGetAction();

	//				printf("option_index=%d\n",option_index);
	//				system("pause");

					switch (option_index+1)
					{
					case 1:
						action_taken=vergence_command[rlmodel4.rlGetAction()];
						break;
		
					case 2:		
						action_taken=vergence_command[rlmodel2.rlGetAction()];
						break;

					case 3:		
						action_taken=vergence_command[rlmodel1.rlGetAction()];
						break;


					default:
						printf("bug\n");
						system("pause");
					}
				
					action_taken_m=vergence_command[rlmodel.rlGetAction()];	


					action_record[iteration]=action_taken;
					action_record_m[iteration]=action_taken_m;
				}	
			
				action_equal=0;
				action_equal_m=0;
				for (int ii=0;ii<iterNum;ii++)
				{
					action_equal+=action_record[ii];
					action_equal_m+=action_record_m[ii];
				}
		
				if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
					action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
				actionEq_record[action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;

				if ((action_equal_m>disparity_range+tolerance) || (action_equal_m<-disparity_range-tolerance))
					action_equal_m=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
				actionEq_record_m[action_equal_m+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
			}

		}
	}
	tomat::push(actionEq_record,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record",0,0);
	tomat::push(actionEq_record_m,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record_m",0,0);


	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","inner_compare_5_3_1.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}

void performance_error_bino_compare2()
{
	const int iterNum=5;
	const int testSize=200;

	int action_record[iterNum] ={0.0};
	int action_record_m[iterNum] ={0.0};

	const int tolerance=100;
	float actionEq_record[TESTNUM*(2*tolerance+1)]={0.0};
	float actionEq_record_m[TESTNUM*(2*tolerance+1)]={0.0};
	

	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error_compare\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('inner_record60.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial3.mat');");

	tomat::get(valWeights,"valw",0,0);
	tomat::get(polWeights,"polw",0,0);

	tomat::get(bases11_m,"b11",0,0);
	tomat::get(bases12_m,"b12",0,0);

	tomat::get(bases21_m,"b21",0,0);
	tomat::get(bases22_m,"b22",0,0);

	tomat::get(bases31_m,"b31",0,0);
	tomat::get(bases32_m,"b32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

/*****************************************************************************************************************/
	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);

	asmodel1_m.setBases(bases11_m,bases12_m);
	asmodel2_m.setBases(bases21_m,bases22_m);
	asmodel3_m.setBases(bases31_m,bases32_m);

	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	rlmodel.setWeights(polWeights,valWeights);


/*****************************************************************************************************************/

	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int lx,ly;
	
	int curDis =-30;	

	for (int frameIndex=0; frameIndex<TESTNUM; frameIndex++)
	{	
		printf("frameIndex=%d\n",frameIndex);

		for(int t=0; t<testSize; t++)
		{
			int i=frameIndex*testSize+t;

			float randPercent=float(rand())/RAND_MAX;
			Image.get_frame_number_together(frameIndex,randPercent,i);
			Image_m.get_frame_number_together(frameIndex,randPercent,i);

			for (int iteration=0; iteration<iterNum; iteration++)
			{
/****************************************************************************************************************************************************/
				Image.window_position_after_action_stereo(curDis, iteration, action_taken, iterNum);
				Image_m.window_position_after_action_stereo(curDis, iteration, action_taken_m, iterNum);
/****************************************************************************************************************************************************/

//				Image.show_image_h();
//				Image_m.show_image_m();
/*
				if (i==0&&iteration==0)
				{
					printf("waiting......");
					while(1)
					{
						Image.show_image_h();
						Image_m.show_image_m();

						if (kbhit())
						{
							if (getch())
								break;
						}
					}
				}
*/

/****************************************************************************************************************************************************/
				FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   	cudaThreadSynchronize();
			    FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();

				FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();
				FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();
				FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();

/****************************************************************************************************************************************************/

				asmodel44.AssomEncode();
				cudaThreadSynchronize();
				asmodel24.AssomEncode();
				cudaThreadSynchronize();
				asmodel22.AssomEncode();
				cudaThreadSynchronize();
				asmodel13.AssomEncode();
				cudaThreadSynchronize();
				asmodel12.AssomEncode();
				cudaThreadSynchronize();
				asmodel11.AssomEncode();
				cudaThreadSynchronize();

				asmodel1_m.AssomEncode();
				cudaThreadSynchronize();	
				asmodel2_m.AssomEncode();
				cudaThreadSynchronize();
				asmodel3_m.AssomEncode();
				cudaThreadSynchronize();

/****************************************************************************************************************************************************/

				option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
				rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
				rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				
				rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
/****************************************************************************************************************************************************/
				rlmodel4.softmaxAct(0);
				rlmodel2.softmaxAct(0);	
				rlmodel1.softmaxAct(0);	
			    option.softmaxAct(0);
			
				rlmodel.softmaxAct(0);

/****************************************************************************************************************************************************/
				rlmodel4.greedyAction();
				rlmodel2.greedyAction();
				rlmodel1.greedyAction();
			    option.greedyAction();
				
				rlmodel.greedyAction();

/****************************************************************************************************************************************************/
				option_index=option.rlGetAction();

//				printf("option_index=%d\n",option_index);
//				system("pause");

				switch (option_index+1)
				{
				case 1:
					action_taken=vergence_command[rlmodel4.rlGetAction()];
					break;
		
				case 2:		
					action_taken=vergence_command[rlmodel2.rlGetAction()];
					break;

				case 3:		
					action_taken=vergence_command[rlmodel1.rlGetAction()];
					break;


				default:
					printf("bug\n");
					system("pause");
				}
				
				action_taken_m=vergence_command[rlmodel.rlGetAction()];	


				action_record[iteration]=action_taken;
				action_record_m[iteration]=action_taken_m;
			}	
			
			action_equal=0;
			action_equal_m=0;
			for (int ii=0;ii<iterNum;ii++)
			{
				action_equal+=action_record[ii];
				action_equal_m+=action_record_m[ii];
			}
	//		printf("action_equal=%d, action_equal_m=%d\n",action_equal,action_equal_m);
		
			if (action_equal>curDis+tolerance)  
				action_equal=curDis+tolerance;
			if (action_equal<curDis-tolerance)  
				action_equal=curDis-tolerance;
		
			actionEq_record[action_equal-curDis+tolerance+(2*tolerance+1)*frameIndex]++;


			if (action_equal_m>curDis+tolerance)  
				action_equal_m=curDis+tolerance;
			if (action_equal_m<curDis-tolerance)  
				action_equal_m=curDis-tolerance;

			actionEq_record_m[action_equal_m-curDis+tolerance+(2*tolerance+1)*frameIndex]++;

	//		printf("action_equal=%d, action_equal_m=%d\n\n",action_equal,action_equal_m);
	//		system("pause");
		}
	}
	tomat::push(actionEq_record,2*tolerance+1,TESTNUM,"actionEq_record",0,0);
	tomat::push(actionEq_record_m,2*tolerance+1,TESTNUM,"actionEq_record_m",0,0);


	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","inner_compare_5_3_2.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}

void performance_error_bino_compare_realTime()
{
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error_compare\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('inner_record40.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial3.mat');");

	tomat::get(valWeights,"valw",0,0);
	tomat::get(polWeights,"polw",0,0);

	tomat::get(bases11_m,"b11",0,0);
	tomat::get(bases12_m,"b12",0,0);

	tomat::get(bases21_m,"b21",0,0);
	tomat::get(bases22_m,"b22",0,0);

	tomat::get(bases31_m,"b31",0,0);
	tomat::get(bases32_m,"b32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

/*****************************************************************************************************************/
	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);

	asmodel1_m.setBases(bases11_m,bases12_m);
	asmodel2_m.setBases(bases21_m,bases22_m);
	asmodel3_m.setBases(bases31_m,bases32_m);

	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	rlmodel.setWeights(polWeights,valWeights);


/*****************************************************************************************************************/
	int last_option_index=0;
	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int frameIndex=0;

	int fovea_disparity=0;;
	int FD_flag=0;
	int display_flag=0;

	int lx=(640-220)/2;
	int ly=(480-220)/2;
	
	int running=1;

	Image.load_image();
	Image_m.load_image();

	while (running)
	{
		if (kbhit())
		{
			switch (getch())
			{
			case 'q':
				running=0;
				break;

			case 'w':
				if (ly>0)
					ly--;
				break;

			case 's':
				if (ly<(480-220-1))
					ly++;
				break;

			case 'a':
				if (lx>0)
					lx--;
				break;

			case 'd':
				if (lx<(640-220-1))
					lx++;
				break;

			case 'r':
				if (frameIndex>180)
					frameIndex=1;
				else
					frameIndex++;
				break;

			case 't':
				if (frameIndex<1)
					frameIndex=179;
				else
					frameIndex--;
				break;

			case 'f':
				fovea_disparity=-20;
				FD_flag=1;
				break;

			case 'g':
				fovea_disparity=20;
				FD_flag=1;
				break;
			
			case 'c':
				display_flag=1;
				break;
			
			case 'v':
				display_flag=0;
				break;
			}
		}
		else
			FD_flag=0;

		Image.get_image_input1(frameIndex,lx, ly, action_taken, fovea_disparity, FD_flag);
		cudaThreadSynchronize();	
		Image_m.get_image_input1(frameIndex,lx, ly, action_taken_m, fovea_disparity, FD_flag); 
		cudaThreadSynchronize();
/****************************************************************************************************************************************************/

		Image.show_image_h(display_flag,option_index);
		Image_m.show_image_m(display_flag);

	

/****************************************************************************************************************************************************/
		FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();

		FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
		cudaThreadSynchronize();
		FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
		cudaThreadSynchronize();
		FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
		cudaThreadSynchronize();

/****************************************************************************************************************************************************/

		asmodel44.AssomEncode();
		cudaThreadSynchronize();
		asmodel24.AssomEncode();
		cudaThreadSynchronize();
		asmodel22.AssomEncode();
		cudaThreadSynchronize();
		asmodel13.AssomEncode();
		cudaThreadSynchronize();
		asmodel12.AssomEncode();
		cudaThreadSynchronize();
		asmodel11.AssomEncode();
		cudaThreadSynchronize();

		asmodel1_m.AssomEncode();
		cudaThreadSynchronize();	
		asmodel2_m.AssomEncode();
		cudaThreadSynchronize();
		asmodel3_m.AssomEncode();
		cudaThreadSynchronize();

/****************************************************************************************************************************************************/

		option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
		rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
		rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
		rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
			
		rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
/****************************************************************************************************************************************************/
		rlmodel4.softmaxAct(0);
		rlmodel2.softmaxAct(0);	
		rlmodel1.softmaxAct(0);	
		option.softmaxAct(0);
			
		rlmodel.softmaxAct(0);

/****************************************************************************************************************************************************/
		rlmodel4.greedyAction();
		rlmodel2.greedyAction();
		rlmodel1.greedyAction();
		option.greedyAction();
				
		rlmodel.greedyAction();

/****************************************************************************************************************************************************/
		last_option_index=option_index;
		option_index=option.rlGetAction();

//		printf("option_index=%d\n",option_index);
//		system("pause");
/*
		switch (last_option_index)
		{
		case 0:
			switch (option_index)
			{
			case 0:
				printf("    |\n");
				break;
			case 1:
				printf("    |____\n");
				break;
			case 2:
				printf("    |________\n");
				break;
			}
			break;
		case 1:
			switch (option_index)
			{
			case 0:
				printf("    ____|\n");
				break;
			case 1:
				printf("        |\n");
				break;
			case 2:
				printf("        |____\n");
				break;
			}
			break;
		case 2:
			switch (option_index)
			{
			case 0:
				printf("    ________|\n");
				break;
			case 1:
				printf("        ____|\n");
				break;
			case 2:
				printf("            |\n");
				break;
			}
			break;
		}
*/
		switch (option_index+1)
		{
		case 1:
			printf("____________coarse\n");
			action_taken=vergence_command[rlmodel4.rlGetAction()];
			break;
		
		case 2:		
			printf("_____middle\n");
			action_taken=vergence_command[rlmodel2.rlGetAction()];
			break;

		case 3:		
			printf("fine\n");
			action_taken=vergence_command[rlmodel1.rlGetAction()];
			break;


		default:
			printf("bug\n");
			system("pause");
		}
				
		action_taken_m=vergence_command[rlmodel.rlGetAction()];	


	}	
			
			
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}

void performance_error_bino_compare_max()
{
	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error_compare2\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('inner_record40.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial3.mat');");

	tomat::get(valWeights,"valw",0,0);
	tomat::get(polWeights,"polw",0,0);

	tomat::get(bases11_m,"b11",0,0);
	tomat::get(bases12_m,"b12",0,0);

	tomat::get(bases21_m,"b21",0,0);
	tomat::get(bases22_m,"b22",0,0);

	tomat::get(bases31_m,"b31",0,0);
	tomat::get(bases32_m,"b32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

/*****************************************************************************************************************/
	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);

	asmodel1_m.setBases(bases11_m,bases12_m);
	asmodel2_m.setBases(bases21_m,bases22_m);
	asmodel3_m.setBases(bases31_m,bases32_m);

	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	rlmodel.setWeights(polWeights,valWeights);


/*****************************************************************************************************************/
	int last_option_index=0;
	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int frameIndex=0;

	int fovea_disparity=0;;
	int FD_flag=0;
	int display_flag=0;
	int new_fixation_flag=0;
	int new_image_flag=0;

	int lx=(640-220)/2;
	int ly=(480-220)/2;
	
	int running=1;

//	Image.load_image();
//	Image_m.load_image();

	while (running)
	{
		if (kbhit())
		{
			switch (getch())
			{
			case 'q':
				running=0;
				break;

			case 'w':
				if (ly>0)
					ly--;
				break;

			case 's':
				if (ly<(480-220-1))
					ly++;
				break;

			case 'a':
				if (lx>0)
					lx--;
				break;

			case 'd':
				if (lx<(640-220-1))
					lx++;
				break;

			case 'r':
				if (frameIndex>=360)
					frameIndex=0;
				else
					frameIndex++;

				new_image_flag=1;
				new_fixation_flag=1;

				break;

			case 't':
				if (frameIndex<0)
					frameIndex=359;
				else
					frameIndex--;

				new_image_flag=1;
				new_fixation_flag=1;

				break;

			case 'f':
				fovea_disparity=-20;
				FD_flag=1;
				break;

			case 'g':
				fovea_disparity=20;
				FD_flag=1;
				break;
			
			case 'c':
				display_flag=1;
				break;
			
			case 'v':
				display_flag=0;
				break;

			case 'e':
				new_fixation_flag=1;
				break;

			case 'z':
				printf("frameIndex=%d*5+1=%d\n",frameIndex,frameIndex*5+1);
				break;
			}
		}
		else
		{
			new_image_flag=0;
			new_fixation_flag=0;
			FD_flag=0;
		}


		if (new_fixation_flag)
		{
			Image.get_fixation_point_max(frameIndex,new_image_flag);
			Image_m.get_fixation_point_max(frameIndex,new_image_flag);
		}

		Image.get_image_input2(frameIndex, action_taken, fovea_disparity, FD_flag);
		cudaThreadSynchronize();	
		Image_m.get_image_input2(frameIndex, action_taken_m, fovea_disparity, FD_flag); 
		cudaThreadSynchronize();

/****************************************************************************************************************************************************/

		Image.show_image_h(display_flag,option_index);
		Image_m.show_image_m(display_flag);

/****************************************************************************************************************************************************/

		FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();
		FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
		cudaThreadSynchronize();

		FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
		cudaThreadSynchronize();
		FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
		cudaThreadSynchronize();
		FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
		cudaThreadSynchronize();

/****************************************************************************************************************************************************/

		asmodel44.AssomEncode();
		cudaThreadSynchronize();
		asmodel24.AssomEncode();
		cudaThreadSynchronize();
		asmodel22.AssomEncode();
		cudaThreadSynchronize();
		asmodel13.AssomEncode();
		cudaThreadSynchronize();
		asmodel12.AssomEncode();
		cudaThreadSynchronize();
		asmodel11.AssomEncode();
		cudaThreadSynchronize();

		asmodel1_m.AssomEncode();
		cudaThreadSynchronize();	
		asmodel2_m.AssomEncode();
		cudaThreadSynchronize();
		asmodel3_m.AssomEncode();
		cudaThreadSynchronize();

/****************************************************************************************************************************************************/

		option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
		rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
		rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
		rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
			
		rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
/****************************************************************************************************************************************************/
		rlmodel4.softmaxAct(0);
		rlmodel2.softmaxAct(0);	
		rlmodel1.softmaxAct(0);	
		option.softmaxAct(0);
			
		rlmodel.softmaxAct(0);

/****************************************************************************************************************************************************/
		rlmodel4.greedyAction();
		rlmodel2.greedyAction();
		rlmodel1.greedyAction();
		option.greedyAction();
				
		rlmodel.greedyAction();

/****************************************************************************************************************************************************/
		last_option_index=option_index;
		option_index=option.rlGetAction();

//		printf("option_index=%d\n",option_index);
//		system("pause");
/*
		switch (last_option_index)
		{
		case 0:
			switch (option_index)
			{
			case 0:
				printf("    |\n");
				break;
			case 1:
				printf("    |____\n");
				break;
			case 2:
				printf("    |________\n");
				break;
			}
			break;
		case 1:
			switch (option_index)
			{
			case 0:
				printf("    ____|\n");
				break;
			case 1:
				printf("        |\n");
				break;
			case 2:
				printf("        |____\n");
				break;
			}
			break;
		case 2:
			switch (option_index)
			{
			case 0:
				printf("    ________|\n");
				break;
			case 1:
				printf("        ____|\n");
				break;
			case 2:
				printf("            |\n");
				break;
			}
			break;
		}
*/
		switch (option_index+1)
		{
		case 1:
		//	printf("____________coarse\n");
			action_taken=vergence_command[rlmodel4.rlGetAction()];
			break;
		
		case 2:		
		//	printf("_____middle\n");
			action_taken=vergence_command[rlmodel2.rlGetAction()];
			break;

		case 3:		
		//	printf("fine\n");
			action_taken=vergence_command[rlmodel1.rlGetAction()];
			break;


		default:
			printf("bug\n");
			system("pause");
		}
				
		action_taken_m=vergence_command[rlmodel.rlGetAction()];	


	}	
			
			
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}




void performance_error_bino_compare_max_result()
{
	const int iterNum=5;
	const int disparity_range=20;
	const int testSize=100;

	int action_record[iterNum] ={0.0};
	int action_record_m[iterNum] ={0.0};

	float error_record[2*disparity_range+1]={0.0};
	float error_record_m[2*disparity_range+1]={0.0};

	const int tolerance=20;
	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};
	float actionEq_record_m[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))]={0.0};
	

	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error_compare3\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('inner_record40.mat');");

	tomat::get(valWeights4,"valw4",0,0);
	tomat::get(valWeights2,"valw2",0,0);
	tomat::get(valWeights1,"valw1",0,0);
	tomat::get(valWeights0,"valw0",0,0);

	tomat::get(polWeights4,"polw4",0,0);
	tomat::get(polWeights2,"polw2",0,0);
	tomat::get(polWeights1,"polw1",0,0);
	tomat::get(polWeights0,"polw0",0,0);

	tomat::get(bases44a,"b44a",0,0);
	tomat::get(bases44b,"b44b",0,0);

	tomat::get(bases24a,"b24a",0,0);
	tomat::get(bases24b,"b24b",0,0);

	tomat::get(bases22a,"b22a",0,0);
	tomat::get(bases22b,"b22b",0,0);

	tomat::get(bases13a,"b13a",0,0);
	tomat::get(bases13b,"b13b",0,0);

	tomat::get(bases12a,"b12a",0,0);
	tomat::get(bases12b,"b12b",0,0);

	tomat::get(bases11a,"b11a",0,0);
	tomat::get(bases11b,"b11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial3.mat');");

	tomat::get(valWeights,"valw",0,0);
	tomat::get(polWeights,"polw",0,0);

	tomat::get(bases11_m,"b11",0,0);
	tomat::get(bases12_m,"b12",0,0);

	tomat::get(bases21_m,"b21",0,0);
	tomat::get(bases22_m,"b22",0,0);

	tomat::get(bases31_m,"b31",0,0);
	tomat::get(bases32_m,"b32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

/*****************************************************************************************************************/
	asmodel44.setBases(bases44a,bases44b);
	asmodel24.setBases(bases24a,bases24b);
	asmodel22.setBases(bases22a,bases22b);
	asmodel13.setBases(bases13a,bases13b);
	asmodel12.setBases(bases12a,bases12b);
	asmodel11.setBases(bases11a,bases11b);

	asmodel1_m.setBases(bases11_m,bases12_m);
	asmodel2_m.setBases(bases21_m,bases22_m);
	asmodel3_m.setBases(bases31_m,bases32_m);

	rlmodel4.setWeights(polWeights4,valWeights4);
	rlmodel2.setWeights(polWeights2,valWeights2);
	rlmodel1.setWeights(polWeights1,valWeights1);
	option.setWeights(polWeights0,valWeights0);

	rlmodel.setWeights(polWeights,valWeights);


/*****************************************************************************************************************/

	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int lx,ly;
	int frameIndex;


	Image.load_image();
	Image_m.load_image();

	int reload=0;
	int newImage=0;

	for(int act=0; act<(2*disparity_range+1); act++)
	{
		int curDis =act-disparity_range;	
		printf("curDis=%d \n",curDis);

		for(int t=0; t<testSize; t++)
		{
			int i=act*testSize+t;
			frameIndex=t/4;


			if (t%4==0)
				newImage=1;
			else
				newImage=0;

			Image.get_fixation_point_max(frameIndex,newImage);
			Image_m.get_fixation_point_max(frameIndex,newImage);


			for (int iteration=0; iteration<iterNum; iteration++)
			{
/****************************************************************************************************************************************************/
				Image.window_position_after_action_stereo(curDis, iteration, action_taken, iterNum);
				Image_m.window_position_after_action_stereo(curDis, iteration, action_taken_m, iterNum);
/****************************************************************************************************************************************************/

/*
				Image.show_image_h(1,option_index);
				Image_m.show_image_m(1);

				if (i==0&&iteration==0)
				{
					printf("waiting......");
					while(1)
					{
						Image.show_image_h();
						Image_m.show_image_m();

						if (kbhit())
						{
							if (getch())
								break;
						}
					}
				}
*/

/****************************************************************************************************************************************************/
				FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   	cudaThreadSynchronize();
			    FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();
			    FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			    cudaThreadSynchronize();

				FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();
				FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();
				FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
				cudaThreadSynchronize();

/****************************************************************************************************************************************************/

				asmodel44.AssomEncode();
				cudaThreadSynchronize();
				asmodel24.AssomEncode();
				cudaThreadSynchronize();
				asmodel22.AssomEncode();
				cudaThreadSynchronize();
				asmodel13.AssomEncode();
				cudaThreadSynchronize();
				asmodel12.AssomEncode();
				cudaThreadSynchronize();
				asmodel11.AssomEncode();
				cudaThreadSynchronize();

				asmodel1_m.AssomEncode();
				cudaThreadSynchronize();	
				asmodel2_m.AssomEncode();
				cudaThreadSynchronize();
				asmodel3_m.AssomEncode();
				cudaThreadSynchronize();

/****************************************************************************************************************************************************/

				option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
			                             ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
				rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
				rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				
				rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
/****************************************************************************************************************************************************/
				rlmodel4.softmaxAct(0);
				rlmodel2.softmaxAct(0);	
				rlmodel1.softmaxAct(0);	
			    option.softmaxAct(0);
			
				rlmodel.softmaxAct(0);

/****************************************************************************************************************************************************/
				rlmodel4.greedyAction();
				rlmodel2.greedyAction();
				rlmodel1.greedyAction();
			    option.greedyAction();
				
				rlmodel.greedyAction();

/****************************************************************************************************************************************************/
				option_index=option.rlGetAction();


//				system("pause");

				switch (option_index+1)
				{
				case 1:
					action_taken=vergence_command[rlmodel4.rlGetAction()];
					break;
		
				case 2:		
					action_taken=vergence_command[rlmodel2.rlGetAction()];
					break;

				case 3:		
					action_taken=vergence_command[rlmodel1.rlGetAction()];
					break;


				default:
					printf("bug\n");
					system("pause");
				}
				
				action_taken_m=vergence_command[rlmodel.rlGetAction()];	


				action_record[iteration]=action_taken;
				action_record_m[iteration]=action_taken_m;
			}	
			
			action_equal=0;
			action_equal_m=0;
			for (int ii=0;ii<iterNum;ii++)
			{
				action_equal+=action_record[ii];
				action_equal_m+=action_record_m[ii];
			}

		/*
			if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
				action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
			actionEq_record[action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;

			if ((action_equal_m>disparity_range+tolerance) || (action_equal_m<-disparity_range-tolerance))
				action_equal_m=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
			actionEq_record_m[action_equal_m+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
		*/

			if (action_equal>disparity_range+tolerance) 
				action_equal=disparity_range+tolerance;
			if (action_equal<-disparity_range-tolerance) 
				action_equal=-disparity_range-tolerance;

			if (action_equal_m>disparity_range+tolerance) 
				action_equal_m=disparity_range+tolerance;
			if (action_equal_m<-disparity_range-tolerance) 
				action_equal_m=-disparity_range-tolerance;

			actionEq_record[action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
			actionEq_record_m[action_equal_m+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
		}

	}

	tomat::push(actionEq_record,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record",0,0);
	tomat::push(actionEq_record_m,2*(disparity_range+tolerance)+1,2*disparity_range+1,"actionEq_record_m",0,0);


	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","inner_compare_5_14_1.mat");
	engEvalString(ep,tmpCmd);
	
	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}



void performance_error_bino_compare_max_result2()
{
	const int checkNum=30;
	const int iterNum=5;
	const int disparity_range=20;
	const int testSize=100;

	int action_record[iterNum] ={0.0};
	int action_record_m[iterNum] ={0.0};

	float error_record[2*disparity_range+1]={0.0};
	float error_record_m[2*disparity_range+1]={0.0};

	const int tolerance=20;

//	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum]={0.0};
//	float actionEq_record_m[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum]={0.0};
	
	float *actionSum_record, *actionSum_record_m;
	actionSum_record=new float [(2*disparity_range+1)*testSize*checkNum];
	actionSum_record_m=new float [(2*disparity_range+1)*testSize*checkNum];


	for (int i=0; i<(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum; i++)
	{
		actionSum_record[i]=0;
		actionSum_record_m[i]=0;
	}

	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM*checkNum);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*checkNum);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3*checkNum);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*checkNum);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*checkNum);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*checkNum);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*checkNum);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\performance_error_compare_bias_debug\\real_PC_side\\Matlab\\saved');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('h2.mat');");

	tomat::get(valWeights4,"valW4",0,0);
	tomat::get(valWeights2,"valW2",0,0);
	tomat::get(valWeights1,"valW1",0,0);
	tomat::get(valWeights0,"valW0",0,0);

	tomat::get(polWeights4,"polW4",0,0);
	tomat::get(polWeights2,"polW2",0,0);
	tomat::get(polWeights1,"polW1",0,0);
	tomat::get(polWeights0,"polW0",0,0);

	tomat::get(bases44a,"B44a",0,0);
	tomat::get(bases44b,"B44b",0,0);

	tomat::get(bases24a,"B24a",0,0);
	tomat::get(bases24b,"B24b",0,0);

	tomat::get(bases22a,"B22a",0,0);
	tomat::get(bases22b,"B22b",0,0);

	tomat::get(bases13a,"B13a",0,0);
	tomat::get(bases13b,"B13b",0,0);

	tomat::get(bases12a,"B12a",0,0);
	tomat::get(bases12b,"B12b",0,0);

	tomat::get(bases11a,"B11a",0,0);
	tomat::get(bases11b,"B11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('m2.mat');");

	tomat::get(valWeights,"valW",0,0);
	tomat::get(polWeights,"polW",0,0);

	tomat::get(bases11_m,"B11",0,0);
	tomat::get(bases12_m,"B12",0,0);

	tomat::get(bases21_m,"B21",0,0);
	tomat::get(bases22_m,"B22",0,0);

	tomat::get(bases31_m,"B31",0,0);
	tomat::get(bases32_m,"B32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

/*****************************************************************************************************************/

	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int lx,ly;
	int frameIndex;


	Image.load_image();
	Image_m.load_image();

	int reload=0;
	int newImage=0;


	for (int checkPoint=0; checkPoint<checkNum; checkPoint++)
	{
		asmodel44.setBases(&bases44a[checkPoint*BASISDIM*BASESNUM],&bases44b[checkPoint*BASISDIM*BASESNUM]);
		asmodel24.setBases(&bases24a[checkPoint*BASISDIM*BASESNUM],&bases24b[checkPoint*BASISDIM*BASESNUM]);
		asmodel22.setBases(&bases22a[checkPoint*BASISDIM*BASESNUM],&bases22b[checkPoint*BASISDIM*BASESNUM]);
		asmodel13.setBases(&bases13a[checkPoint*BASISDIM*BASESNUM],&bases13b[checkPoint*BASISDIM*BASESNUM]);
		asmodel12.setBases(&bases12a[checkPoint*BASISDIM*BASESNUM],&bases12b[checkPoint*BASISDIM*BASESNUM]);
		asmodel11.setBases(&bases11a[checkPoint*BASISDIM*BASESNUM],&bases11b[checkPoint*BASISDIM*BASESNUM]);

		asmodel1_m.setBases(&bases11_m[checkPoint*BASISDIM*BASESNUM],&bases12_m[checkPoint*BASISDIM*BASESNUM]);
		asmodel2_m.setBases(&bases21_m[checkPoint*BASISDIM*BASESNUM],&bases22_m[checkPoint*BASISDIM*BASESNUM]);
		asmodel3_m.setBases(&bases31_m[checkPoint*BASISDIM*BASESNUM],&bases32_m[checkPoint*BASISDIM*BASESNUM]);

		rlmodel4.setWeights(&polWeights4[checkPoint*FEATUREDIM*ACTIONNUM],&valWeights4[checkPoint*FEATUREDIM]);
		rlmodel2.setWeights(&polWeights2[checkPoint*2*FEATUREDIM*ACTIONNUM],&valWeights2[checkPoint*2*FEATUREDIM]);
		rlmodel1.setWeights(&polWeights1[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights1[checkPoint*3*FEATUREDIM]);
	
		option.setWeights(&polWeights0[checkPoint*FEATUREDIM*6*SCALENUM],&valWeights0[checkPoint*FEATUREDIM*6]);

		rlmodel.setWeights(&polWeights[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights[checkPoint*3*FEATUREDIM]);


		printf("\n\ncheckPoint=%d\n",checkPoint);

		for(int act=0; act<(2*disparity_range+1); act++)
		{
			int curDis =act-disparity_range;	
			printf("%d ",curDis);

			for(int t=0; t<testSize; t++)
			{
				int i=act*testSize+t;
				frameIndex=t/4;


				if (t%4==0)
					newImage=1;
				else
					newImage=0;

				Image.get_fixation_point_max(frameIndex,newImage);
				Image_m.get_fixation_point_max(frameIndex,newImage);


				for (int iteration=0; iteration<iterNum; iteration++)
				{
	/****************************************************************************************************************************************************/
					Image.window_position_after_action_stereo(curDis, iteration, action_taken, iterNum);
					Image_m.window_position_after_action_stereo(curDis, iteration, action_taken_m, iterNum);
	/****************************************************************************************************************************************************/

	/*
					Image.show_image_h(1,option_index);
					Image_m.show_image_m(1);

					if (i==0&&iteration==0)
					{
						printf("waiting......");
						while(1)
						{
							Image.show_image_h();
							Image_m.show_image_m();

							if (kbhit())
							{
								if (getch())
									break;
							}
						}
					}
	*/

	/****************************************************************************************************************************************************/
					FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   		cudaThreadSynchronize();
					FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();

					FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();
					FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();
					FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();

	/****************************************************************************************************************************************************/

					asmodel44.AssomEncode();
					cudaThreadSynchronize();
					asmodel24.AssomEncode();
					cudaThreadSynchronize();
					asmodel22.AssomEncode();
					cudaThreadSynchronize();
					asmodel13.AssomEncode();
					cudaThreadSynchronize();
					asmodel12.AssomEncode();
					cudaThreadSynchronize();
					asmodel11.AssomEncode();
					cudaThreadSynchronize();

					asmodel1_m.AssomEncode();
					cudaThreadSynchronize();	
					asmodel2_m.AssomEncode();
					cudaThreadSynchronize();
					asmodel3_m.AssomEncode();
					cudaThreadSynchronize();

	/****************************************************************************************************************************************************/

					option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
											 ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
					rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
					rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				
					rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
	/****************************************************************************************************************************************************/
					rlmodel4.softmaxAct(0);
					rlmodel2.softmaxAct(0);	
					rlmodel1.softmaxAct(0);	
					option.softmaxAct(0);
			
					rlmodel.softmaxAct(0);

	/****************************************************************************************************************************************************/
					option.greedyAction();
		
	/****************************************************************************************************************************************************/
					option_index=option.rlGetAction();


	//				system("pause");

					switch (option_index+1)
					{
					case 1:
						action_taken=vergence_command[rlmodel4.rlGetAction()];
						break;
		
					case 2:		
						action_taken=vergence_command[rlmodel2.rlGetAction()];
						break;

					case 3:		
						action_taken=vergence_command[rlmodel1.rlGetAction()];
						break;


					default:
						printf("bug\n");
						system("pause");
					}
				
					action_taken_m=vergence_command[rlmodel.rlGetAction()];	


					action_record[iteration]=action_taken;
					action_record_m[iteration]=action_taken_m;
				}	
			
				action_equal=0;
				action_equal_m=0;
				for (int ii=0;ii<iterNum;ii++)
				{
					action_equal+=action_record[ii];
					action_equal_m+=action_record_m[ii];
				}

				actionSum_record[act+t*(2*disparity_range+1)+checkPoint*(2*disparity_range+1)*testSize]=action_equal;
				actionSum_record_m[act+t*(2*disparity_range+1)+checkPoint*(2*disparity_range+1)*testSize]=action_equal_m;
				
			}
		}

		tomat::push(actionSum_record,(2*disparity_range+1)*testSize,checkNum,"actionEq_record",0,0);
		tomat::push(actionSum_record_m,(2*disparity_range+1)*testSize,checkNum,"actionEq_record_m",0,0);


		char tmpCmd[100];

		sprintf(tmpCmd,"save('%s')","trial2_compare_internalRecord.mat");
		engEvalString(ep,tmpCmd);
	}

	tomat::push(actionSum_record,(2*disparity_range+1)*testSize,checkNum,"actionEq_record",0,0);
	tomat::push(actionSum_record_m,(2*disparity_range+1)*testSize,checkNum,"actionEq_record_m",0,0);


	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","trial2_compare_result.mat");
	engEvalString(ep,tmpCmd);
	



	free(actionSum_record);
	free(actionSum_record_m);

	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}


void performance_error_bino_compare_max_result3()
{
	const int checkNum=30;
	const int iterNum=5;
	const int disparity_range=20;
	const int testSize=100;

	int action_record[iterNum] ={0.0};
	int action_record_m[iterNum] ={0.0};

	float error_record[2*disparity_range+1]={0.0};
	float error_record_m[2*disparity_range+1]={0.0};

	const int tolerance=20;

//	float actionEq_record[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum]={0.0};
//	float actionEq_record_m[(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum]={0.0};
	
	float *actionEq_record, *actionEq_record_m;
	actionEq_record=new float [(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum];
	actionEq_record_m=new float [(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum];

	for (int i=0; i<(2*disparity_range+1)*((2*(disparity_range+tolerance)+1))*checkNum; i++)
	{
		actionEq_record[i]=0;
		actionEq_record_m[i]=0;
	}

	float *bases44a, *bases44b;
	float *bases24a, *bases24b;
	float *bases22a, *bases22b;
	float *bases13a, *bases13b;
	float *bases12a, *bases12b;
	float *bases11a, *bases11b;

	float *bases11_m, *bases12_m;
	float *bases21_m, *bases22_m;
	float *bases31_m, *bases32_m;

	float *valWeights, *polWeights;

	float *valWeights4, *polWeights4;
	float *valWeights2, *polWeights2;
	float *valWeights1, *polWeights1;
	float *valWeights0, *polWeights0;

	float* filter;

	srand (5);
/*****************************************************************************************************************/

	filter= (float*)malloc(sizeof(float)*FILT_WIDTH*FILT_WIDTH);

	bases44a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases44b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases24b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases13b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11a = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases11b = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	bases11_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases12_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases21_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases22_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases31_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);
	bases32_m = (float*)malloc(sizeof(float)*BASISDIM*BASESNUM*checkNum);

	polWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*ACTIONNUM*checkNum);
	valWeights = (float*)malloc(sizeof(float)*FEATUREDIM*3*checkNum);


	polWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*ACTIONNUM*checkNum);
	polWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*3*checkNum);

	valWeights4 = (float*)malloc(sizeof(float)*FEATUREDIM*checkNum);
	valWeights2 = (float*)malloc(sizeof(float)*2*FEATUREDIM*checkNum);
	valWeights1 = (float*)malloc(sizeof(float)*3*FEATUREDIM*checkNum);
	valWeights0 = (float*)malloc(sizeof(float)*6*FEATUREDIM*checkNum);

/*****************************************************************************************************************/

	tomat::start();

	engEvalString(ep,"cd('D:\\zzt\\Mphil\\code\\new_dataset_saliency_together_elegant_performance_error_compare3\\real_PC_side\\Matlab\\saved\\trial1');");

	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial1_hiera.mat');");

	tomat::get(valWeights4,"valW4",0,0);
	tomat::get(valWeights2,"valW2",0,0);
	tomat::get(valWeights1,"valW1",0,0);
	tomat::get(valWeights0,"valW0",0,0);

	tomat::get(polWeights4,"polW4",0,0);
	tomat::get(polWeights2,"polW2",0,0);
	tomat::get(polWeights1,"polW1",0,0);
	tomat::get(polWeights0,"polW0",0,0);

	tomat::get(bases44a,"B44a",0,0);
	tomat::get(bases44b,"B44b",0,0);

	tomat::get(bases24a,"B24a",0,0);
	tomat::get(bases24b,"B24b",0,0);

	tomat::get(bases22a,"B22a",0,0);
	tomat::get(bases22b,"B22b",0,0);

	tomat::get(bases13a,"B13a",0,0);
	tomat::get(bases13b,"B13b",0,0);

	tomat::get(bases12a,"B12a",0,0);
	tomat::get(bases12b,"B12b",0,0);

	tomat::get(bases11a,"B11a",0,0);
	tomat::get(bases11b,"B11b",0,0);



	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('trial1_multi.mat');");

	tomat::get(valWeights,"valW",0,0);
	tomat::get(polWeights,"polW",0,0);

	tomat::get(bases11_m,"B11",0,0);
	tomat::get(bases12_m,"B12",0,0);

	tomat::get(bases21_m,"B21",0,0);
	tomat::get(bases22_m,"B22",0,0);

	tomat::get(bases31_m,"B31",0,0);
	tomat::get(bases32_m,"B32",0,0);


	engEvalString(ep,"clear all;");
	engEvalString(ep,"load('initParam.mat');");
	tomat::get(filter,"filter55",0,0);

/*****************************************************************************************************************/


	ImageLoader Image;
	ImageLoader Image_m;


	BatchInput FoveaBatch44(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch24(4, 2, filter);  // (10+2*9)*4=112
	BatchInput FoveaBatch22(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch13(3, 1, filter);  // (10+1*9)*3=57
	BatchInput FoveaBatch12(2, 2, filter);  // (10+2*9)*2=56
	BatchInput FoveaBatch11(1, 5, filter);  // (10+5*9)*1=55

	BatchInput FoveaBatch1_m(4, 5, filter);  // (10+5*9)*4=220
	BatchInput FoveaBatch2_m(2, 5, filter);  // (10+5*9)*2=110
	BatchInput FoveaBatch3_m(1, 5, filter);  // (10+5*9)*1=55


	AssomOnline asmodel44(FoveaBatch44.getBatch());
	AssomOnline asmodel24(FoveaBatch24.getBatch());
	AssomOnline asmodel22(FoveaBatch22.getBatch());
	AssomOnline asmodel13(FoveaBatch13.getBatch());
	AssomOnline asmodel12(FoveaBatch12.getBatch());
	AssomOnline asmodel11(FoveaBatch11.getBatch());

	AssomOnline asmodel1_m(FoveaBatch1_m.getBatch());
	AssomOnline asmodel2_m(FoveaBatch2_m.getBatch());
	AssomOnline asmodel3_m(FoveaBatch3_m.getBatch());



	ReinforcementLearner rlmodel4(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM,ACTIONNUM);
	ReinforcementLearner rlmodel2(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*2,ACTIONNUM);
	ReinforcementLearner rlmodel1(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);
	ReinforcementLearner option(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*6,SCALENUM);

	ReinforcementLearner rlmodel(ALPHA_V2,ALPHA_N2,ALPHA_P2,FEATUREDIM*3,ACTIONNUM);

/*****************************************************************************************************************/

	int option_index=0;
	int action_taken=0;
	int action_taken_m=0;
	int action_equal=0;
	int action_equal_m=0;
	int lx,ly;
	int frameIndex;


	Image.load_image();
	Image_m.load_image();

	int reload=0;
	int newImage=0;


	for (int checkPoint=0; checkPoint<checkNum; checkPoint++)
	{
		asmodel44.setBases(&bases44a[checkPoint*BASISDIM*BASESNUM],&bases44b[checkPoint*BASISDIM*BASESNUM]);
		asmodel24.setBases(&bases24a[checkPoint*BASISDIM*BASESNUM],&bases24b[checkPoint*BASISDIM*BASESNUM]);
		asmodel22.setBases(&bases22a[checkPoint*BASISDIM*BASESNUM],&bases22b[checkPoint*BASISDIM*BASESNUM]);
		asmodel13.setBases(&bases13a[checkPoint*BASISDIM*BASESNUM],&bases13b[checkPoint*BASISDIM*BASESNUM]);
		asmodel12.setBases(&bases12a[checkPoint*BASISDIM*BASESNUM],&bases12b[checkPoint*BASISDIM*BASESNUM]);
		asmodel11.setBases(&bases11a[checkPoint*BASISDIM*BASESNUM],&bases11b[checkPoint*BASISDIM*BASESNUM]);

		asmodel1_m.setBases(&bases11_m[checkPoint*BASISDIM*BASESNUM],&bases12_m[checkPoint*BASISDIM*BASESNUM]);
		asmodel2_m.setBases(&bases21_m[checkPoint*BASISDIM*BASESNUM],&bases22_m[checkPoint*BASISDIM*BASESNUM]);
		asmodel3_m.setBases(&bases31_m[checkPoint*BASISDIM*BASESNUM],&bases32_m[checkPoint*BASISDIM*BASESNUM]);

		rlmodel4.setWeights(&polWeights4[checkPoint*FEATUREDIM*ACTIONNUM],&valWeights4[checkPoint*FEATUREDIM]);
		rlmodel2.setWeights(&polWeights2[checkPoint*2*FEATUREDIM*ACTIONNUM],&valWeights2[checkPoint*2*FEATUREDIM]);
		rlmodel1.setWeights(&polWeights1[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights1[checkPoint*3*FEATUREDIM]);
	
		option.setWeights(&polWeights0[checkPoint*FEATUREDIM*6*SCALENUM],&valWeights0[checkPoint*FEATUREDIM*6]);

		rlmodel.setWeights(&polWeights[checkPoint*3*FEATUREDIM*ACTIONNUM],&valWeights[checkPoint*3*FEATUREDIM]);


		printf("\n\ncheckPoint=%d\n",checkPoint);

		for(int act=0; act<(2*disparity_range+1); act++)
		{
			int curDis =act-disparity_range;	
			printf("%d ",curDis);

			for(int t=0; t<testSize; t++)
			{
				int i=act*testSize+t;
				frameIndex=t/4;


				if (t%4==0)
					newImage=1;
				else
					newImage=0;

				float randPercent=float(rand())/RAND_MAX;

				Image.get_fixation_point_saliency(frameIndex,newImage,randPercent);
				Image_m.get_fixation_point_saliency(frameIndex,newImage,randPercent);


				for (int iteration=0; iteration<iterNum; iteration++)
				{
	/****************************************************************************************************************************************************/
					Image.window_position_after_action_mono(curDis, iteration, action_taken, iterNum);
					Image_m.window_position_after_action_mono(curDis, iteration, action_taken_m, iterNum);
	/****************************************************************************************************************************************************/

/*
					Image.show_image_h_mono();
					Image_m.show_image_m_mono();

					if (i==0&&iteration==0)
					{
						printf("waiting......");
						while(1)
						{
							Image.show_image_h_mono();
							Image_m.show_image_m_mono();

							if (kbhit())
							{
								if (getch())
									break;
							}
						}
					}
	
					system("pause");
*/

	/****************************************************************************************************************************************************/
					FoveaBatch44.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
			   		cudaThreadSynchronize();
					FoveaBatch24.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch22.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch13.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch12.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();
					FoveaBatch11.get_batch_input(Image.leftWin,Image.rightWin, Image.left_x_pos,Image.right_x_pos,Image.y_pos);
					cudaThreadSynchronize();

					FoveaBatch1_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();
					FoveaBatch2_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();
					FoveaBatch3_m.get_batch_input(Image_m.leftWin,Image_m.rightWin, Image_m.left_x_pos,Image_m.right_x_pos,Image_m.y_pos);
					cudaThreadSynchronize();

	/****************************************************************************************************************************************************/

					asmodel44.AssomEncode();
					cudaThreadSynchronize();
					asmodel24.AssomEncode();
					cudaThreadSynchronize();
					asmodel22.AssomEncode();
					cudaThreadSynchronize();
					asmodel13.AssomEncode();
					cudaThreadSynchronize();
					asmodel12.AssomEncode();
					cudaThreadSynchronize();
					asmodel11.AssomEncode();
					cudaThreadSynchronize();

					asmodel1_m.AssomEncode();
					cudaThreadSynchronize();	
					asmodel2_m.AssomEncode();
					cudaThreadSynchronize();
					asmodel3_m.AssomEncode();
					cudaThreadSynchronize();

	/****************************************************************************************************************************************************/

					option.cudaGetFeatureRewardAssom(asmodel44.getCoef(),asmodel24.getCoef(),asmodel22.getCoef(),asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef()
											 ,asmodel44.getResidue(),asmodel24.getResidue(),asmodel22.getResidue(),asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
					rlmodel4.cudaGetFeatureRewardAssom(asmodel44.getCoef(), asmodel44.getResidue());
					rlmodel2.cudaGetFeatureRewardAssom(asmodel24.getCoef(),asmodel22.getCoef(), asmodel24.getResidue(),asmodel22.getResidue());
					rlmodel1.cudaGetFeatureRewardAssom(asmodel13.getCoef(),asmodel12.getCoef(),asmodel11.getCoef(), asmodel13.getResidue(),asmodel12.getResidue(),asmodel11.getResidue());
				
					rlmodel.cudaGetFeatureRewardAssom(asmodel1_m.getCoef(),asmodel2_m.getCoef(),asmodel3_m.getCoef(), asmodel1_m.getResidue(),asmodel2_m.getResidue(),asmodel3_m.getResidue());
		
	/****************************************************************************************************************************************************/
					rlmodel4.softmaxAct(0);
					rlmodel2.softmaxAct(0);	
					rlmodel1.softmaxAct(0);	
					option.softmaxAct(0);
			
					rlmodel.softmaxAct(0);

	/****************************************************************************************************************************************************/
					rlmodel4.greedyAction();
					rlmodel2.greedyAction();
					rlmodel1.greedyAction();
					option.greedyAction();
				
					rlmodel.greedyAction();

	/****************************************************************************************************************************************************/
					option_index=option.rlGetAction();


	//				system("pause");

					switch (option_index+1)
					{
					case 1:
						action_taken=vergence_command[rlmodel4.rlGetAction()];
						break;
		
					case 2:		
						action_taken=vergence_command[rlmodel2.rlGetAction()];
						break;

					case 3:		
						action_taken=vergence_command[rlmodel1.rlGetAction()];
						break;


					default:
						printf("bug\n");
						system("pause");
					}
				
					action_taken_m=vergence_command[rlmodel.rlGetAction()];	


					action_record[iteration]=action_taken;
					action_record_m[iteration]=action_taken_m;
				}	
			
				action_equal=0;
				action_equal_m=0;
				for (int ii=0;ii<iterNum;ii++)
				{
					action_equal+=action_record[ii];
					action_equal_m+=action_record_m[ii];
				}

			
				if ((action_equal>disparity_range+tolerance) || (action_equal<-disparity_range-tolerance))
					action_equal=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
		//		actionEq_record[action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;

				if ((action_equal_m>disparity_range+tolerance) || (action_equal_m<-disparity_range-tolerance))
					action_equal_m=rand()%(2*(disparity_range+tolerance)+1)-(disparity_range+tolerance);
		//		actionEq_record_m[action_equal_m+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act]++;
			
			/*
				if (action_equal>disparity_range+tolerance) 
					action_equal=disparity_range+tolerance;
				if (action_equal<-disparity_range-tolerance) 
					action_equal=-disparity_range-tolerance;

				if (action_equal_m>disparity_range+tolerance) 
					action_equal_m=disparity_range+tolerance;
				if (action_equal_m<-disparity_range-tolerance) 
					action_equal_m=-disparity_range-tolerance;
			*/
				int _index_=action_equal+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act+(2*(disparity_range+tolerance)+1)*(2*disparity_range+1)*checkPoint;
				actionEq_record[_index_]++;

				int _index_m=action_equal_m+(disparity_range+tolerance)+(2*(disparity_range+tolerance)+1)*act+(2*(disparity_range+tolerance)+1)*(2*disparity_range+1)*checkPoint;
				actionEq_record_m[_index_m]++;
			}
		}

		tomat::push(actionEq_record,(2*(disparity_range+tolerance)+1)*(2*disparity_range+1),checkNum,"actionEq_record",0,0);
		tomat::push(actionEq_record_m,(2*(disparity_range+tolerance)+1)*(2*disparity_range+1),checkNum,"actionEq_record_m",0,0);


		char tmpCmd[100];
		sprintf(tmpCmd,"save('%s')","trial1_compare_internalRecord.mat");
		engEvalString(ep,tmpCmd);
	}

	tomat::push(actionEq_record,(2*(disparity_range+tolerance)+1)*(2*disparity_range+1),checkNum,"actionEq_record",0,0);
	tomat::push(actionEq_record_m,(2*(disparity_range+tolerance)+1)*(2*disparity_range+1),checkNum,"actionEq_record_m",0,0);


	char tmpCmd[100];
	sprintf(tmpCmd,"save('%s')","trial1_compare_result_mono.mat");
	engEvalString(ep,tmpCmd);
	



	free(actionEq_record);
	free(actionEq_record_m);

	free(bases44a);
	free(bases44b);
	free(bases24a);
	free(bases24b);
	free(bases22a);
	free(bases22b);
	free(bases13a);
	free(bases13b);
	free(bases12a);
	free(bases12b);
	free(bases11a);
	free(bases11b);

	free(valWeights4);
	free(valWeights2);
	free(valWeights1);
	free(valWeights0);

	free(polWeights4);
	free(polWeights2);
	free(polWeights1);
	free(polWeights0);


	free(bases11_m);
	free(bases12_m);
	free(bases21_m);
	free(bases22_m);
	free(bases31_m);
	free(bases32_m);

	free(valWeights);
	free(polWeights);
}