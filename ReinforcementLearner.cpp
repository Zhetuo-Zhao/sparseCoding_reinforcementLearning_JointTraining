#include "ReinforcementLearner.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include "ToMatlab.h"

ReinforcementLearner::ReinforcementLearner(): J(0),taken(5),lastError(1),alpha_v(ALPHA_V), alpha_n(ALPHA_N), alpha_p(ALPHA_P), inputDim(FEATUREDIM), outputDim(ACTIONNUM)
{
	actions = new float[outputDim];
	
	for(int i = 0; i < outputDim; i++)
		actions[i] = (i-5)*0.1;


	hostPolicy = new float[outputDim];

	cudaMalloc((void**) &policyWeights,	sizeof(float)*inputDim*outputDim);
	cudaMalloc((void**) &valueWeights,	sizeof(float)*inputDim);
	cudaMalloc((void**) &w,	sizeof(float)*inputDim*outputDim);
	cudaMalloc((void**) &feature,	sizeof(float)*inputDim);
	cudaMalloc((void**) &feature_lastState,	sizeof(float)*inputDim);

	cudaMalloc((void**) &devicePolicy,	sizeof(float)*outputDim);
	cudaMalloc((void**) &devicePolicy_softmax,	sizeof(float)*outputDim);

	cudaMalloc((void**) &policy_lastState_selection,	sizeof(float)*outputDim);
	cudaMalloc((void**) &softmax_policy,	sizeof(float)*outputDim);
	cudaMalloc((void**) &psi,	sizeof(float)*inputDim*outputDim);
	prevCommands[0]=0;prevCommands[1]=0;

	initWeights();
	float *hostW = new float[inputDim*outputDim];
	for(int i = 0; i < inputDim*outputDim; i++)
	{
		hostW[i] = 0;
	}
	cudaMemcpy(w,hostW,sizeof(float)*inputDim*outputDim,cudaMemcpyHostToDevice);

	float *hostF = new float[inputDim];
	for(int i = 0; i < inputDim; i++)
	{
		hostF[i] = float(rand())/float(INT_MAX);
	}
	cudaMemcpy(feature_lastState,hostF,sizeof(float)*inputDim,cudaMemcpyHostToDevice);

	hostGreedyPol=new int[outputDim];
	float *hostP = new float[inputDim];
	for(int i = 0; i < outputDim; i++)
	{
		hostP[i] = float(rand())/float(INT_MAX);
		hostGreedyPol[i]=0;
	}
	cudaMemcpy(policy_lastState_selection,hostP,sizeof(float)*outputDim,cudaMemcpyHostToDevice);

	updateFlag=false;
	count=0;

	//initialize values 
	value_lastState=float(rand())/float(INT_MAX);	
	cudaMemset (softmax_policy,0,outputDim*sizeof(float));
	lastErr=1;

	errMweight_host= new float [AGENTSNUM];
	cudaMalloc((void**) &errMweight_device,	sizeof(float)*AGENTSNUM);

	float errMsum=0;
	for (int iy=0;iy<10;iy++)
	{
		for (int ix=0;ix<10;ix++)
		{
			errMweight_host[iy*10+ix]=exp(-(ix-4.5)*(ix-4.5)/36-(iy-4.5)*(iy-4.5)/36);
			errMsum+=errMweight_host[iy*10+ix];
		}
	}

	for (int iii=0; iii<AGENTSNUM; iii++)
		errMweight_host[iii]/=errMsum;

	cudaMemcpy(errMweight_device,errMweight_host,sizeof(float)*AGENTSNUM,cudaMemcpyHostToDevice);


	delete[] hostW;
	delete[] hostF;
	delete[] hostP;
}



ReinforcementLearner::ReinforcementLearner(float vv, float nn, float pp, int inD, int outD)
{
	J=0;
	taken=(outD-1)/2;
	lastError=1;
	alpha_v=vv;
	alpha_n=nn;
	alpha_p=pp;

	inputDim=inD;
	outputDim=outD;

	actions = new float[outputDim];
	
	for(int i = 0; i < outputDim; i++)
		actions[i] = (i-5)*0.1;


	hostPolicy = new float[outputDim];



	cudaMalloc((void**) &policyWeights,	sizeof(float)*inputDim*outputDim);
	cudaMalloc((void**) &valueWeights,	sizeof(float)*inputDim);
	cudaMalloc((void**) &w,	sizeof(float)*inputDim*outputDim);
	cudaMalloc((void**) &feature,	sizeof(float)*inputDim);
	cudaMalloc((void**) &feature_lastState,	sizeof(float)*inputDim);

	cudaMalloc((void**) &devicePolicy,	sizeof(float)*outputDim);

	cudaMalloc((void**) &policy_lastState_selection,	sizeof(float)*outputDim);
	cudaMalloc((void**) &softmax_policy,	sizeof(float)*outputDim);
	cudaMalloc((void**) &psi,	sizeof(float)*inputDim*outputDim);
	prevCommands[0]=0;prevCommands[1]=0;

	initWeights();
	float *hostW = new float[inputDim*outputDim];
	for(int i = 0; i < inputDim*outputDim; i++)
	{
		hostW[i] = 0;
	}
	cudaMemcpy(w,hostW,sizeof(float)*inputDim*outputDim,cudaMemcpyHostToDevice);

	float *hostF = new float[inputDim];
	for(int i = 0; i < inputDim; i++)
	{
		hostF[i] = float(rand())/float(INT_MAX);
	}
	cudaMemcpy(feature_lastState,hostF,sizeof(float)*inputDim,cudaMemcpyHostToDevice);

	hostGreedyPol=new int[outputDim];
	float *hostP = new float[inputDim];
	for(int i = 0; i < outputDim; i++)
	{
		hostP[i] = float(rand())/float(INT_MAX);
		hostGreedyPol[i]=0;
	}
	cudaMemcpy(policy_lastState_selection,hostP,sizeof(float)*outputDim,cudaMemcpyHostToDevice);

	updateFlag=false;
	count=0;

	//initialize values 
	value_lastState=float(rand())/float(INT_MAX);	
	cudaMemset (softmax_policy,0,outputDim*sizeof(float));
	lastErr=1;



	errMweight_host= new float [AGENTSNUM];
	cudaMalloc((void**) &errMweight_device,	sizeof(float)*AGENTSNUM);

	float errMsum=0;
	for (int iy=0;iy<10;iy++)
	{
		for (int ix=0;ix<10;ix++)
		{
			errMweight_host[iy*10+ix]=exp(-(ix-4.5)*(ix-4.5)/36-(iy-4.5)*(iy-4.5)/36);
			errMsum+=errMweight_host[iy*10+ix];
		}
	}

	for (int iii=0; iii<AGENTSNUM; iii++)
		errMweight_host[iii]/=errMsum;

	cudaMemcpy(errMweight_device,errMweight_host,sizeof(float)*AGENTSNUM,cudaMemcpyHostToDevice);


	delete[] hostW;
	delete[] hostF;
	delete[] hostP;
}




ReinforcementLearner::~ReinforcementLearner()
{
	delete[] errMweight_host;
	cudaFree(errMweight_device);

	delete[] actions;
	delete[] hostPolicy;
	delete[] hostGreedyPol;
	cudaFree(policyWeights);
	cudaFree(feature_lastState);
	cudaFree(valueWeights);
	cudaFree(w);
	cudaFree(softmax_policy);
	cudaFree(feature);
	cudaFree(psi);
}
/*
void ReinforcementLearner::cudaGetFeatureReward(float* coef,float* residue)
{
	float error;
	float ratio = 0.1;

	cublasSdgmm(cuhandle,CUBLAS_SIDE_LEFT,FEATUREDIM,1,coef,FEATUREDIM,coef,1,feature,FEATUREDIM);
	//cublasSscal(cuhandle,FEATUREDIM-1,&ratio,feature,1);
	cublasSdot(cuhandle,BASISDIM*AGENTSNUM,residue,1,residue,1,&error);
	error = error/100; //mean error
	reward = (lastError-error+EPS)/(lastError+error+EPS);
	lastError = error;
	
	//reward = -error;
}
*/
void ReinforcementLearner::cudaGetFeatureRewardAssom(float* coef,float* residue)
{
	float error;
	
	cudaMemcpy(feature,coef,sizeof(float)*(inputDim),cudaMemcpyDeviceToDevice); 
//	cublasSdot(cuhandle,AGENTSNUM,errMweight_device,1,residue,1,&error);
	cublasSasum(cuhandle,AGENTSNUM,residue,1,&error);
	error=error/AGENTSNUM;

#if AVRGREWARD
	float avrErr=error;
	reward = (lastErr-avrErr+EPS)/(lastErr+avrErr+EPS);
	lastErr=avrErr;
#else
	reward = -error; //tnc original
//	printf("reward=%f\n",reward);
#endif
}

void ReinforcementLearner::cudaGetFeatureRewardAssom(float* coef1,float* coef2,float* residue1,float* residue2)
{
	float error1, error2, error3;

	//feature=coef;
	cudaMemcpy(feature,coef1,sizeof(float)*(inputDim/2),cudaMemcpyDeviceToDevice); 
	cudaMemcpy(&feature[inputDim*1/2],coef2,sizeof(float)*(inputDim/2),cudaMemcpyDeviceToDevice);

	cublasSasum(cuhandle,AGENTSNUM,residue1,1,&error1);
	cublasSasum(cuhandle,AGENTSNUM,residue2,1,&error2);

	error1=error1/AGENTSNUM;
	error2=error2/AGENTSNUM;

#if AVRGREWARD
	float avrErr= (error2+error1)/2;

	reward = (lastErr-avrErr+EPS)/(lastErr+avrErr+EPS);
	lastErr=avrErr;
#else
	reward = -1*(error2+error1)/2; //tnc original
#endif
}

void ReinforcementLearner::cudaGetFeatureRewardAssom(float* coef1,float* coef2,float* coef3,float* residue1,float* residue2,float* residue3)
{
	float error1, error2, error3;

	//feature=coef;
	cudaMemcpy(feature,coef1,sizeof(float)*(inputDim/3),cudaMemcpyDeviceToDevice); 
	cudaMemcpy(&feature[inputDim*1/3],coef2,sizeof(float)*(inputDim/3),cudaMemcpyDeviceToDevice);
	cudaMemcpy(&feature[inputDim*2/3],coef3,sizeof(float)*(inputDim/3),cudaMemcpyDeviceToDevice);

	cublasSasum(cuhandle,AGENTSNUM,residue1,1,&error1);
	cublasSasum(cuhandle,AGENTSNUM,residue2,1,&error2);
	cublasSasum(cuhandle,AGENTSNUM,residue3,1,&error3);

	error1=error1/AGENTSNUM;
	error2=error2/AGENTSNUM;
	error3=error3/AGENTSNUM;

#if AVRGREWARD
//	float avrErr= (error4+error3/2+error2/3+error1/4)/(AGENTSNUM*(1+0.5+0.25+0.333));
	float avrErr= (error3+error2+error1)/3;
//	float avrErr= error4/AGENTSNUM;

	reward = (lastErr-avrErr+EPS)/(lastErr+avrErr+EPS);
	lastErr=avrErr;
#else
	reward = -1*(error3+error2+error1)/3; //tnc original
#endif
}

void ReinforcementLearner::cudaGetFeatureRewardAssom(float* coef44,float* coef24,float* coef22,float* coef13,float* coef12,float* coef11
	                                                ,float* residue44,float* residue24,float* residue22,float* residue13,float* residue12,float* residue11)
{
	float error44, error24, error22, error13, error12, error11;
	float error4, error2, error1;

	//feature=coef;
	cudaMemcpy(feature,coef44,sizeof(float)*(inputDim/6),cudaMemcpyDeviceToDevice); 
	cudaMemcpy(&feature[inputDim*1/6],coef24,sizeof(float)*(inputDim/6),cudaMemcpyDeviceToDevice);
	cudaMemcpy(&feature[inputDim*2/6],coef22,sizeof(float)*(inputDim/6),cudaMemcpyDeviceToDevice);
	cudaMemcpy(&feature[inputDim*3/6],coef13,sizeof(float)*(inputDim/6),cudaMemcpyDeviceToDevice);
	cudaMemcpy(&feature[inputDim*4/6],coef12,sizeof(float)*(inputDim/6),cudaMemcpyDeviceToDevice);
	cudaMemcpy(&feature[inputDim*5/6],coef11,sizeof(float)*(inputDim/6),cudaMemcpyDeviceToDevice);

	cublasSasum(cuhandle,AGENTSNUM,residue44,1,&error44);
	cublasSasum(cuhandle,AGENTSNUM,residue24,1,&error24);
	cublasSasum(cuhandle,AGENTSNUM,residue22,1,&error22);
	cublasSasum(cuhandle,AGENTSNUM,residue13,1,&error13);
	cublasSasum(cuhandle,AGENTSNUM,residue12,1,&error12);
	cublasSasum(cuhandle,AGENTSNUM,residue11,1,&error11);

	error4=error44/AGENTSNUM;
	error2=(error24/2+error22/2)/AGENTSNUM;
	error1=(error13/3+error12/3+error11/3)/AGENTSNUM;


#if AVRGREWARD
	float avrErr= (error4+error2+error1)/3;

	reward = (lastErr-avrErr+EPS)/(lastErr+avrErr+EPS);
	lastErr=avrErr;
#else
	reward = -1*(error4+error2+error1)/3; //tnc original
#endif
}


void ReinforcementLearner::updatePrevCommands(float command)
{
	prevCommands[1]=prevCommands[0];
	prevCommands[0]=command;
}

void ReinforcementLearner::softmaxAct(int flag)
{
	cublasSgemv(cuhandle,CUBLAS_OP_N,outputDim,inputDim,&cublasOne,policyWeights,outputDim,feature,1,&cublasZero,devicePolicy,1);
	cudaMemcpy(hostPolicy,devicePolicy,sizeof(float)*outputDim,cudaMemcpyDeviceToHost);
//	printf("host: %f, %f, %f\n", hostPolicy[0], hostPolicy[1], hostPolicy[2]);									  
	hostSoftMaxPolicy(flag);
}


void ReinforcementLearner::greedyAction()
{
	float m = 0;
	int m_index=0;
	for(int i = 0; i < outputDim; i++)
	{
		if (m<hostPolicy[i])
		{
			m=hostPolicy[i];
			m_index=i;
		}
	}

	for (int i=0;i<outputDim;i++)
	{
		hostGreedyPol[i]=0;
	}
	taken = m_index;
	hostGreedyPol[taken]=1;

}


void ReinforcementLearner::hostSoftMaxPolicy(int flag)
{
	float m = 0;
	float sumExpPolicy = 0;
	for(int i = 0; i < outputDim; i++)
	{
		m = max(hostPolicy[i],m);
	}
	for(int i = 0; i < outputDim; i++)
	{
		hostPolicy[i] = exp(hostPolicy[i]-m);
		sumExpPolicy += hostPolicy[i];
	}
	for(int i = 0; i < outputDim; i++)
	{
		hostPolicy[i] /= sumExpPolicy;
	}
	//tomat::push(hostPolicy,ACTIONNUM,1,"hostpolicy",0,0);

	float randSelection = float(rand())/RAND_MAX;
	for(int i = 0; i < outputDim; i++)
	{
		if(randSelection < hostPolicy[i])
		{
			taken = i;
			break;
		}else
			randSelection -= hostPolicy[i];
	}
	
	if (flag)
	{
		float* policy_selection;
		policy_selection=new float [outputDim];

		for(int i = 0; i < outputDim; i++)
		{
			policy_selection[i] = -hostPolicy[i];
		}
		policy_selection[taken] += 1;
		cudaMemcpy(policy_lastState_selection,policy_selection,sizeof(float)*outputDim,cudaMemcpyHostToDevice);
		updatePrevCommands(float(taken)/(outputDim-1)-0.5);

		delete [] policy_selection;
		//cudaMemcpy(softmax_policy,policy_selection,sizeof(float)*ACTIONNUM,cudaMemcpyHostToDevice);
	}
}



void ReinforcementLearner::updateNetwork(bool isToUpdate)
{
	if(isToUpdate)
	{
		float value;
		cublasSdot(cuhandle,inputDim,feature,1,valueWeights,1,&value);
		
		J = (1-GAMMA)*J + GAMMA*reward;
//		printf("J=%f\n",J);
		float delta = reward - J + XI*value - value_lastState;
		float ratio = delta * alpha_v;

		cublasSaxpy(cuhandle,inputDim,&ratio,feature_lastState,1,valueWeights,1);
		
		cublasSgemm(cuhandle,CUBLAS_OP_N,CUBLAS_OP_N,outputDim,inputDim,1,&cublasOne,policy_lastState_selection,outputDim,feature_lastState,1,&cublasZero,psi,outputDim);

		float adv;
		cublasSdot(cuhandle,outputDim*inputDim,psi,1,w,1,&adv);
		adv = (delta - adv)*alpha_n;
		cublasSaxpy(cuhandle,outputDim*inputDim,&adv,psi,1,w,1);

		float decay = 1 - alpha_p*LAMBDA;
		cublasSscal(cuhandle,outputDim*inputDim,&decay,policyWeights,1);
		cublasSaxpy(cuhandle,outputDim*inputDim,&alpha_p,w,1,policyWeights,1);
		
		//tomat::push(policyWeights,ACTIONNUM,FEATUREDIM,"policyWeightsAfter");
	}
	cublasSdot(cuhandle,inputDim,feature,1,valueWeights,1,&value_lastState);
	softmaxAct(1);
	cudaMemcpy(feature_lastState,feature,sizeof(float)*inputDim,cudaMemcpyDeviceToDevice);
}

void ReinforcementLearner::updateNetwork(bool isToUpdate, int last_taken)
{
	if(isToUpdate)
	{
		float* policy_selection;
		policy_selection=new float [outputDim];

		for(int i = 0; i < outputDim; i++)
		{
			policy_selection[i] = -hostPolicy[i];
		}
		policy_selection[last_taken] += 1;

		cudaMemcpy(policy_lastState_selection,policy_selection,sizeof(float)*outputDim,cudaMemcpyHostToDevice);
		updatePrevCommands(float(last_taken)/(outputDim-1)-0.5);
		delete [] policy_selection;


		float value;
		cublasSdot(cuhandle,inputDim,feature,1,valueWeights,1,&value);
		
		J = (1-GAMMA)*J + GAMMA*reward;
		float delta = reward - J + XI*value - value_lastState;
		float ratio = delta * alpha_v;

		cublasSaxpy(cuhandle,inputDim,&ratio,feature_lastState,1,valueWeights,1);
		
		cublasSgemm(cuhandle,CUBLAS_OP_N,CUBLAS_OP_N,outputDim,inputDim,1,&cublasOne,policy_lastState_selection,outputDim,feature_lastState,1,&cublasZero,psi,outputDim);

		float adv;
		cublasSdot(cuhandle,outputDim*inputDim,psi,1,w,1,&adv);
		adv = (delta - adv)*alpha_n;
		cublasSaxpy(cuhandle,outputDim*inputDim,&adv,psi,1,w,1);

		float decay = 1 - alpha_p*LAMBDA;
		cublasSscal(cuhandle,outputDim*inputDim,&decay,policyWeights,1);
		cublasSaxpy(cuhandle,outputDim*inputDim,&alpha_p,w,1,policyWeights,1);
		
		//tomat::push(policyWeights,ACTIONNUM,FEATUREDIM,"policyWeightsAfter");
	}
	cublasSdot(cuhandle,inputDim,feature,1,valueWeights,1,&value_lastState);
	softmaxAct(0);
	cudaMemcpy(feature_lastState,feature,sizeof(float)*inputDim,cudaMemcpyDeviceToDevice);
}

void ReinforcementLearner::updateNetwork(bool isToUpdate, int last_taken, float prob)
{
	if(isToUpdate)
	{
		float alpha_v_prob=alpha_v*prob;
		float alpha_p_prob=alpha_p*prob;
		float alpha_n_prob=alpha_n*prob;

		float* policy_selection;
		policy_selection=new float [outputDim];

		for(int i = 0; i < outputDim; i++)
		{
			policy_selection[i] = -hostPolicy[i];
		}
		policy_selection[last_taken] += 1;

		cudaMemcpy(policy_lastState_selection,policy_selection,sizeof(float)*outputDim,cudaMemcpyHostToDevice);
		updatePrevCommands(float(last_taken)/(outputDim-1)-0.5);
		delete [] policy_selection;


		float value;
		cublasSdot(cuhandle,inputDim,feature,1,valueWeights,1,&value);
		
		J = (1-GAMMA)*J + GAMMA*reward;
		float delta = reward - J + XI*value - value_lastState;
		float ratio = delta * alpha_v_prob;

		cublasSaxpy(cuhandle,inputDim,&ratio,feature_lastState,1,valueWeights,1);
		
		cublasSgemm(cuhandle,CUBLAS_OP_N,CUBLAS_OP_N,outputDim,inputDim,1,&cublasOne,policy_lastState_selection,outputDim,feature_lastState,1,&cublasZero,psi,outputDim);

		float adv;
		cublasSdot(cuhandle,outputDim*inputDim,psi,1,w,1,&adv);
		adv = (delta - adv)*alpha_n_prob;
		cublasSaxpy(cuhandle,outputDim*inputDim,&adv,psi,1,w,1);

		float decay = 1 - alpha_p_prob*LAMBDA;
		cublasSscal(cuhandle,outputDim*inputDim,&decay,policyWeights,1);
		cublasSaxpy(cuhandle,outputDim*inputDim,&alpha_p_prob,w,1,policyWeights,1);
		
		//tomat::push(policyWeights,ACTIONNUM,FEATUREDIM,"policyWeightsAfter");
	}
	cublasSdot(cuhandle,inputDim,feature,1,valueWeights,1,&value_lastState);
	softmaxAct(0);
	cudaMemcpy(feature_lastState,feature,sizeof(float)*inputDim,cudaMemcpyDeviceToDevice);
}

float* ReinforcementLearner::getPWeights()
{
	return policyWeights;
}

float* ReinforcementLearner::getNWeights()
{
	return w;
}

float* ReinforcementLearner::getVWeights()
{
	return valueWeights;
}

float ReinforcementLearner::getAverageReward()
{
	return J;
}




float* ReinforcementLearner::getHostPolicy()
{
	return hostPolicy;
}

float* ReinforcementLearner::getReward()
{
	return &reward;
}

float* ReinforcementLearner::getFeature()
{
	return feature;
}

void ReinforcementLearner::setAlpha(float v, float n, float p)
{
	alpha_v = v;
	alpha_p = p;
	alpha_n = n;
}

void ReinforcementLearner::initWeights(float* hostPWeights,float* hostVWeights)
{
		
	cudaMemcpy(valueWeights,hostVWeights,sizeof(float)*inputDim,cudaMemcpyHostToDevice);
	cudaMemcpy(policyWeights,hostPWeights,sizeof(float)*inputDim*outputDim,cudaMemcpyHostToDevice);
	
}

void ReinforcementLearner::setWeights(float*hostpWeights, float*hostvWeights)
{
	cudaMemcpy(valueWeights,hostvWeights,sizeof(float)*inputDim,cudaMemcpyHostToDevice);
	cudaMemcpy(policyWeights,hostpWeights,sizeof(float)*inputDim*outputDim,cudaMemcpyHostToDevice);
}

void ReinforcementLearner::setWeights(float* hostPWeights,float* hostVWeights, float* hostNWeight, float hostJ)
{
		
	cudaMemcpy(valueWeights,hostVWeights,sizeof(float)*inputDim,cudaMemcpyHostToDevice);
	cudaMemcpy(policyWeights,hostPWeights,sizeof(float)*inputDim*outputDim,cudaMemcpyHostToDevice);
	cudaMemcpy(w,hostNWeight,sizeof(float)*inputDim*outputDim,cudaMemcpyHostToDevice);

	J=hostJ;
}


void ReinforcementLearner::initWeights()
{
	float *hostPolicyWeights = new float[inputDim*outputDim];
	for(int i = 0; i < inputDim*outputDim; i++)
	{
		hostPolicyWeights[i] = (float(rand())/RAND_MAX-0.5)*INITIALWEIGHTSRANGE1*2;
	}
	cudaMemcpy(policyWeights,hostPolicyWeights,sizeof(float)*inputDim*outputDim,cudaMemcpyHostToDevice);

	float *hostValueWeights = new float[inputDim];
	for(int i = 0; i < inputDim; i++)
	{
		hostValueWeights[i] = (float(rand())/RAND_MAX-0.5)*INITIALWEIGHTSRANGE2*2;
	}
	cudaMemcpy(valueWeights,hostValueWeights,sizeof(float)*inputDim,cudaMemcpyHostToDevice);

	delete[] hostPolicyWeights;
	delete[] hostValueWeights;
}

unsigned int ReinforcementLearner::rlGetAction()
{
	return taken; //default is 5 which is 0 //tnc
}

void ReinforcementLearner::updateNetwork()
{
	
		float value;
		cublasSdot(cuhandle,inputDim,feature,1,valueWeights,1,&value);

		J = (1-GAMMA)*J + GAMMA*reward;
//		printf("J=%f\n",J);

		float delta = reward - J + XI*value - value_lastState;
		
		float ratio = delta * alpha_v;
		cublasSaxpy(cuhandle,inputDim,&ratio,feature_lastState,1,valueWeights,1);

		cublasSgemm(cuhandle,CUBLAS_OP_N,CUBLAS_OP_N,outputDim,inputDim,1,&cublasOne,policy_lastState_selection,outputDim,feature_lastState,1,&cublasZero,psi,outputDim);

		float adv;
		cublasSdot(cuhandle,outputDim*inputDim,psi,1,w,1,&adv);
		adv = (delta - adv)*alpha_n;
		cublasSaxpy(cuhandle,outputDim*inputDim,&adv,psi,1,w,1);


		float decay = 1 - alpha_p*LAMBDA;
		cublasSscal(cuhandle,outputDim*inputDim,&decay,policyWeights,1);
		cublasSaxpy(cuhandle,outputDim*inputDim,&alpha_p,w,1,policyWeights,1);


		//set value_lastState and policy_lastState
		cudaMemcpy(feature_lastState,feature,sizeof(float)*inputDim,cudaMemcpyDeviceToDevice); 
		cudaMemcpy(policy_lastState_selection,softmax_policy,sizeof(float)*outputDim,cudaMemcpyDeviceToDevice); 
		value_lastState = value;
		//cublasSdot(cuhandle,FEATUREDIM,feature,1,valueWeights,1,&value_lastState);

}


void ReinforcementLearner::updateLastProb(float* lastProb)
{
	lastProb[0]=hostPolicy[0];
	lastProb[1]=hostPolicy[1];
	lastProb[2]=hostPolicy[2];
	lastProb[3]=hostPolicy[3];
}

