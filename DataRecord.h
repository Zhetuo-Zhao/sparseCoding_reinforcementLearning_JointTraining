
#pragma once
#include "config.h"

extern int seed;

class DataRecord
{
private:
	bool created;

public:

	int* checkPoints;
	int currCp;
	int tTrain;

	DataRecord();
	~DataRecord();

	void create(int a);

	void saveBases44(float* b1,float* b2 );
	void saveBases24(float* b1,float* b2 );
	void saveBases22(float* b1,float* b2 );
	void saveBases13(float* b1,float* b2 );
	void saveBases12(float* b1,float* b2 );
	void saveBases11(float* b1,float* b2 );

	void saveWeights4(float* pWeight, float* vWeight, float* nWeight);
	void saveWeights2(float* pWeight, float* vWeight, float* nWeight);
	void saveWeights1(float* pWeight, float* vWeight, float* nWeight);
	void saveWeights0(float* pWeight, float* vWeight, float* nWeight);

	void saveOption(float* o);

	void saveToMat();

	float *bases44a, *bases24a, *bases22a, *bases13a, *bases12a, *bases11a;
	float *bases44b, *bases24b, *bases22b, *bases13b, *bases12b, *bases11b;

	float *policyWeights4, *policyWeights2, *policyWeights1, *policyWeights0;
	float *valueWeights4, *valueWeights2, *valueWeights1, *valueWeights0;
	float *ww4, *ww2, *ww1, *ww0;


	
};

extern DataRecord drecord;
