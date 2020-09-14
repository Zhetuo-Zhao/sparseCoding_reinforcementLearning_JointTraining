#include "AssomOnline.h"
#include "ToMatlab.h"
#include <cmath>
#include <cuda_runtime.h>





AssomOnline::AssomOnline(float* batchInput)
	:residue(batchInput),eta(ETA)
{
	cudaMalloc((void**) &bases1,	sizeof(float)*BASESNUM*BASISDIM);
	cudaMalloc((void**) &bases2,	sizeof(float)*BASESNUM*BASISDIM);
	cudaMalloc((void**) &coef,	sizeof(float)*(BASESNUM+1));
	cudaMalloc((void**) &corrBX1,	sizeof(float)*BASESNUM*AGENTSNUM);
	cudaMalloc((void**) &corrBX2,	sizeof(float)*BASESNUM*AGENTSNUM);
	cudaMalloc((void**) &corrTmp1,	sizeof(float)*BASESNUM*AGENTSNUM);
	cudaMalloc((void**) &corrTmp2,	sizeof(float)*BASESNUM*AGENTSNUM); 
	cudaMalloc((void**) &probTmp1,	sizeof(float)*BASESNUM*BASESNUM*AGENTSNUM);
	cudaMalloc((void**) &proj,	sizeof(float)*BASESNUM*AGENTSNUM); 
	cudaMalloc((void**) &emission,	sizeof(float)*BASESNUM*AGENTSNUM); 
	cudaMalloc((void**) &index,	sizeof(int)*BASESNUM*AGENTSNUM);
	cudaMalloc((void**) &winners,	sizeof(int)*AGENTSNUM);
	cudaMalloc((void**) &winErr,	sizeof(float)*AGENTSNUM);
	cudaMalloc((void**) &wInputTmp1,	sizeof(float)*3*BASESNUM);
	cudaMalloc((void**) &wInputTmp2,	sizeof(float)*BASISDIM*AGENTSNUM*BASESNUM); 
	cudaMalloc((void**) &wInputTmp3,	sizeof(float)*BASISDIM*BASESNUM);
	cudaMalloc((void**) &wInputTmp4,	sizeof(float)*BASISDIM*BASESNUM); 
	cudaMalloc((void**) &diff1,	sizeof(float)*BASISDIM*BASESNUM);
	cudaMalloc((void**) &diff2,	sizeof(float)*BASISDIM*BASESNUM); 
	cudaMalloc((void**) &tmpBases1,	sizeof(float)*BASISDIM*BASESNUM);
	cudaMalloc((void**) &tmpBases2,	sizeof(float)*BASISDIM*BASESNUM); 
	cudaMalloc((void**) &nodeProb,	sizeof(float)*BASESNUM*AGENTSNUM);
	cudaMalloc((void**) &transProb,	sizeof(float)*BASESNUM*BASESNUM);
	cudaMalloc((void**) &ones,	sizeof(float)*AGENTSNUM*1);
	cudaMalloc((void**) &winProj,	sizeof(float)*AGENTSNUM*1);

	cudaMalloc((void**) &weightedBases,	sizeof(float)*BASISDIM*AGENTSNUM);
	
	float *hostRandBases1 = new float[BASESNUM*BASISDIM];
	float *hostRandBases2 = new float[BASESNUM*BASISDIM];
	float *hostRandNodeProb = new float[BASESNUM*AGENTSNUM];
	float *hostTransitionProb = new float[BASESNUM*BASESNUM];
	float *hostOnes = new float[AGENTSNUM];

	//initililize basis vectors
	for(int i = 0; i < BASESNUM*BASISDIM; i++)
	{
		hostRandBases1[i] = float(rand())/RAND_MAX;
		hostRandBases2[i] = float(rand())/RAND_MAX;
	}
	cudaMemcpy(bases1,hostRandBases1,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyHostToDevice);
	cudaMemcpy(bases2,hostRandBases2,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyHostToDevice);

	//tomat::push(bases1,BASISDIM,BASESNUM,"bases",1,0);
	
//	float *t1;
//	t1=(float*)malloc(BASISDIM*BASESNUM*sizeof(float));
//	cudaMemcpy(t1,bases2,sizeof(float)*BASISDIM*BASESNUM,cudaMemcpyDeviceToHost);
	cudaNormalizeBases();


	//initialize probability
	for(int i = 0; i < BASESNUM*AGENTSNUM; i++)
	{
		hostRandNodeProb[i] = float(rand())/RAND_MAX;
	}
	cudaMemcpy(nodeProb,hostRandNodeProb,sizeof(float)*BASESNUM*AGENTSNUM,cudaMemcpyHostToDevice);
	
	cudaNormalizeProb();
	//tomat::push(nodeProb,BASESNUM,AGENTSNUM,"np",1,0);

	//initialize the transition probability table 
	hostGenTransitionProb(hostTransitionProb,1.25,0.2);
	cudaMemcpy(transProb,hostTransitionProb,sizeof(float)*BASESNUM*BASESNUM,cudaMemcpyHostToDevice);

	float bias = 0.001;
	cudaMemcpy(&coef[BASESNUM],&bias,sizeof(float),cudaMemcpyHostToDevice);

	for(int i=0; i<AGENTSNUM; i++) hostOnes[i]=1.0; 
	cudaMemcpy(ones,hostOnes,sizeof(float)*AGENTSNUM,cudaMemcpyHostToDevice);
	

	normSigmaW = 1/(2*SIGMA_W*SIGMA_W);
	normSigmaN = 1/(2*SIGMA_N*SIGMA_N);

	cudaThreadSynchronize();
//	tomat::push(transProb,BASESNUM,BASESNUM,"tp",1,0);
//	tomat::push(hostTransitionProb,BASESNUM,BASESNUM,"tph",0,0);

	residueHost = new float[AGENTSNUM];
	winProjHost = new float[AGENTSNUM];

	iter=0;
	delete[] hostRandBases1;
	delete[] hostRandBases2;
	delete[] hostRandNodeProb;
	delete[] hostOnes;
}

AssomOnline::~AssomOnline()
{
	cudaFree(bases1);
	cudaFree(bases2);
	cudaFree(coef);
	cudaFree(corrBX1);
	cudaFree(corrBX2);
	cudaFree(corrTmp1);
	cudaFree(corrTmp2);
	cudaFree(proj);
	cudaFree(emission);
	cudaFree(probTmp1);
	cudaFree(weightedBases);
	cudaFree(index);
	cudaFree(winners);
	cudaFree(winErr);
	cudaFree(wInputTmp1);
	cudaFree(wInputTmp2);
	cudaFree(wInputTmp3);
	cudaFree(wInputTmp4);
	cudaFree(diff1);
	cudaFree(diff2);
	cudaFree(tmpBases1);
	cudaFree(tmpBases2);
	cudaFree(transProb);
	cudaFree(nodeProb);
	cudaFree(ones);
	cudaFree(winProj);

	delete[] residueHost;
	delete[] winProjHost;
}

void AssomOnline::AssomEncode()
{
	cudaAssomEncode();
}

void AssomOnline::hostGenTransitionProb(float* hostTransitionProb, float sigma, float alpha)
{

	float sums[BASESNUM];
	float pUni = (float)1.0/(float)BASESNUM;
	float var = pow(sigma,2);
	
	for (int row=0; row< TOPO_SUBSPACE; row++)
	{
        for( int col=0; col<TOPO_SUBSPACE; col++)
        {   
			int rownum= row+col*TOPO_SUBSPACE;
			sums[rownum]=0.0;
                      
           for (int el=0; el<BASESNUM; el++)
		   {	
              //[x y] = ind2sub(topo, el);
			  int x = el/TOPO_SUBSPACE;	
			  int y = el%TOPO_SUBSPACE;
              hostTransitionProb[rownum+el*BASESNUM] = exp(-0.5*( pow((float)(x-row),2) + pow((float)(y-col),2))/var);              
			  sums[rownum] += hostTransitionProb[rownum+el*BASESNUM];
           }
           
		}
            
	}

	for(int i=0; i<BASESNUM; i++)
	{
		for(int j=0; j<BASESNUM; j++)
		{
			hostTransitionProb[i+j*BASESNUM] = hostTransitionProb[i+j*BASESNUM]/sums[i];
		}
	}


}

void AssomOnline::updateBases()
{
	cudaUpdateBases();
}

float* AssomOnline::getCoef()
{
	return coef;
}

float AssomOnline::getErr()
{
	float error;
	cublasSasum(cuhandle,AGENTSNUM,winErr,1,&error);
	return error/AGENTSNUM;
}

float* AssomOnline::getResidue()
{
	return winErr;
}


float* AssomOnline::getResidueHost()
{
	cudaMemcpy(residueHost,winErr,sizeof(float)*AGENTSNUM,cudaMemcpyDeviceToHost);
	return residueHost;
}

float* AssomOnline::getWinProjHost()
{
	cudaMemcpy(winProjHost,winProj,sizeof(float)*AGENTSNUM,cudaMemcpyDeviceToHost);
	return winProjHost;
}


void AssomOnline::setBases(float*hostBases1, float*hostBases2)
{
	cudaMemcpy(bases1,hostBases1,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyHostToDevice);
	cudaMemcpy(bases2,hostBases2,sizeof(float)*BASESNUM*BASISDIM,cudaMemcpyHostToDevice);
}