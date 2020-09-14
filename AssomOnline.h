#pragma once
#include "config.h"
class AssomOnline
{
private:

	float* coef;
	float* residue;
	float* corrBX1;
	float* corrBX2;
	float* corrTmp1;
	float* corrTmp2;
	float* probTmp1;
	int *index;
	float* proj;
	float* emission;
	float* weightedBases;
	float* transProb;
	float *residueHost;
	float *winProj;
	float *winProjHost;
	
	float eta;
	int* winners;
	float *wInputTmp1;
	float *wInputTmp2;
	float *wInputTmp3;
	float *wInputTmp4;
	float *diff1;
	float *diff2;
	float *tmpBases1;
	float *tmpBases2;
	float *winErr;
	int iter;
	float normSigmaW;
	float normSigmaN;
	float *ones;

	void cudaAssomEncode();
	void cudaNormalizeBases();
	void cudaNormalizeProb();
	void cudaUpdateBases();

public:
	float* bases1;
	float* bases2;
	float* nodeProb;

public:
	AssomOnline(){}
	~AssomOnline();
	AssomOnline(float* batchInput);
	void AssomEncode();
	void updateBases();
	float* getBases();
	float* getCoef();
	float* getResidue();
	float* getResidueHost();
	float getErr();
	void hostGenTransitionProb(float* hostTransitionProb, float sigma, float alpha);
	void setBases(float*hostBases1, float*hostBases2);
	float* getWinProjHost();
};
