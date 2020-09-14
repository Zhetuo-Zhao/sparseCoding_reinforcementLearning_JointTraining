#pragma once
#include "config.h"

#define AVRGREWARD 1

class ReinforcementLearner
{
private:
	float alpha_v;
	float alpha_p;
	float alpha_n;

	int inputDim;
	int outputDim;

	float* policyWeights;
	float* valueWeights;
	float J;
	float* w;

	float* psi;

	float* feature;
	float reward;
	float lastErr;
	float lastError;
	float* devicePolicy;
	float* devicePolicy_softmax;

	float prevCommands[2];

	float* feature_lastState;
	float value_lastState;
	float* policy_lastState_selection;
	float* softmax_policy;
	bool updateFlag; 
	unsigned int count;

	void hostSoftMaxPolicy(int flag);

	float* errMweight_host;
	float* errMweight_device;

public:
	float* actions;
	unsigned int taken;

	float* hostPolicy;

	int* hostGreedyPol;

	ReinforcementLearner();
	ReinforcementLearner(float vv, float nn, float pp, int inD, int outD);

	~ReinforcementLearner();
	void initWeights(float* pWeights,float* vWeights);
	void initWeights();
	float* getPWeights();
	float* getVWeights();
	float* getNWeights();
	float getAverageReward();


	float* getHostPolicy();
	float* getReward();
	float* getFeature();
	void greedyAction();

	void cudaGetFeature(float* coef);
//	void cudaGetFeatureReward(float* coef,float* residue);
	void cudaGetFeatureRewardAssom(float* coef1, float* coef2, float* coef3, float* coef4, float* coef5, float* coef6
		                          ,float* residue1, float* residue2, float* residue3,float* residue4, float* residue5, float* residue6);

	void cudaGetFeatureRewardAssom(float* coef1, float* coef2, float* coef3
		                          ,float* residue1, float* residue2, float* residue3);

	void cudaGetFeatureRewardAssom(float* coef1, float* coef2
		                          ,float* residue1, float* residue2);

	void cudaGetFeatureRewardAssom(float* coef,float* residue);
	


	unsigned int rlGetAction();

	void softmaxAct(int flag);

	void updateNetwork(bool isToUpdate); //old function which did both updating and action generation
	void updateNetwork(bool isToUpdate, int last_taken);
	void updateNetwork(bool isToUpdate, int last_taken, float prob);

	void setAlpha(float v, float n, float p);
	void setWeights(float*pWeights, float*vWeights);
	void setWeights(float* pWeights,float* vWeights, float* nWeights, float j);

	void updateNetwork(); // new function which only does the updating
	void updatePrevCommands(float command);
	void updateLastProb(float* lastProb);
};
