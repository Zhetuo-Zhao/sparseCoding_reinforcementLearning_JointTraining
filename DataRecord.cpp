#include "DataRecord.h"
#include <cuda_runtime.h>
#include"ToMatlab.h"

DataRecord drecord;

DataRecord::DataRecord()
{
	
}

void DataRecord::create(int maxIter)
{
	currCp = 0;
	tTrain = maxIter;
	checkPoints = new int[CHECKPOINTSNUM];
	for(int i = 0; i < CHECKPOINTSNUM; i++)
	{
		checkPoints[i] = (tTrain-1)*i/(CHECKPOINTSNUM-1);
	}

	bases44a = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases44b = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases24a = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases24b = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases22a = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases22b = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases13a = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases13b = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases12a = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases12b = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases11a = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];
	bases11b = new float[BASESNUM*BASISDIM*CHECKPOINTSNUM];

	policyWeights4 = new float[FEATUREDIM*ACTIONNUM*CHECKPOINTSNUM];
	policyWeights2 = new float[2*FEATUREDIM*ACTIONNUM*CHECKPOINTSNUM];
	policyWeights1 = new float[3*FEATUREDIM*ACTIONNUM*CHECKPOINTSNUM];
	policyWeights0 = new float[6*FEATUREDIM*SCALENUM*CHECKPOINTSNUM];

	valueWeights4 = new float[FEATUREDIM*CHECKPOINTSNUM];
	valueWeights2 = new float[2*FEATUREDIM*CHECKPOINTSNUM];
	valueWeights1 = new float[3*FEATUREDIM*CHECKPOINTSNUM];
	valueWeights0 = new float[6*FEATUREDIM*CHECKPOINTSNUM];

	ww4 = new float[FEATUREDIM*ACTIONNUM*CHECKPOINTSNUM];
	ww2 = new float[2*FEATUREDIM*ACTIONNUM*CHECKPOINTSNUM];
	ww1 = new float[3*FEATUREDIM*ACTIONNUM*CHECKPOINTSNUM];
	ww0 = new float[6*FEATUREDIM*SCALENUM*CHECKPOINTSNUM];

	
}



void DataRecord::saveBases44(float* b1,float* b2 )
{
	cudaMemcpy(&bases44a[currCp*BASESNUM*BASISDIM],b1,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&bases44b[currCp*BASESNUM*BASISDIM],b2,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveBases24(float* b1,float* b2 )
{
	cudaMemcpy(&bases24a[currCp*BASESNUM*BASISDIM],b1,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&bases24b[currCp*BASESNUM*BASISDIM],b2,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveBases22(float* b1,float* b2 )
{
	cudaMemcpy(&bases22a[currCp*BASESNUM*BASISDIM],b1,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&bases22b[currCp*BASESNUM*BASISDIM],b2,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveBases13(float* b1,float* b2 )
{
	cudaMemcpy(&bases13a[currCp*BASESNUM*BASISDIM],b1,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&bases13b[currCp*BASESNUM*BASISDIM],b2,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveBases12(float* b1,float* b2 )
{
	cudaMemcpy(&bases12a[currCp*BASESNUM*BASISDIM],b1,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&bases12b[currCp*BASESNUM*BASISDIM],b2,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveBases11(float* b1,float* b2 )
{
	cudaMemcpy(&bases11a[currCp*BASESNUM*BASISDIM],b1,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&bases11b[currCp*BASESNUM*BASISDIM],b2,BASESNUM*BASISDIM*sizeof(float),cudaMemcpyDeviceToHost);
}





void DataRecord::saveWeights4(float* pWeight, float* vWeight, float* nWeight)
{
	cudaMemcpy(&policyWeights4[currCp*FEATUREDIM*ACTIONNUM],pWeight,FEATUREDIM*ACTIONNUM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&valueWeights4[currCp*FEATUREDIM],vWeight,FEATUREDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&ww4[currCp*FEATUREDIM*ACTIONNUM],nWeight,FEATUREDIM*ACTIONNUM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveWeights2(float* pWeight, float* vWeight, float* nWeight)
{
	cudaMemcpy(&policyWeights2[currCp*2*FEATUREDIM*ACTIONNUM],pWeight,2*FEATUREDIM*ACTIONNUM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&valueWeights2[currCp*2*FEATUREDIM],vWeight,2*FEATUREDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&ww2[currCp*2*FEATUREDIM*ACTIONNUM],nWeight,2*FEATUREDIM*ACTIONNUM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveWeights1(float* pWeight, float* vWeight, float* nWeight)
{
	cudaMemcpy(&policyWeights1[currCp*3*FEATUREDIM*ACTIONNUM],pWeight,3*FEATUREDIM*ACTIONNUM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&valueWeights1[currCp*3*FEATUREDIM],vWeight,3*FEATUREDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&ww1[currCp*3*FEATUREDIM*ACTIONNUM],nWeight,3*FEATUREDIM*ACTIONNUM*sizeof(float),cudaMemcpyDeviceToHost);
}

void DataRecord::saveWeights0(float* pWeight, float* vWeight, float* nWeight)
{
	cudaMemcpy(&policyWeights0[currCp*6*FEATUREDIM*SCALENUM],pWeight,6*FEATUREDIM*SCALENUM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&valueWeights0[currCp*6*FEATUREDIM],vWeight,6*FEATUREDIM*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&ww0[currCp*6*FEATUREDIM*SCALENUM],nWeight,6*FEATUREDIM*SCALENUM*sizeof(float),cudaMemcpyDeviceToHost);
}



DataRecord::~DataRecord()
{
	if(created)
	{
		delete[] checkPoints;
	
		delete[] bases44a;
		delete[] bases44b;
		delete[] bases24a;
		delete[] bases24b;
		delete[] bases22a;
		delete[] bases22b;
		delete[] bases13a;
		delete[] bases13b;
		delete[] bases12a;
		delete[] bases12b;
		delete[] bases11a;
		delete[] bases11b;

		delete[] policyWeights4;
		delete[] policyWeights2;
		delete[] policyWeights1;
		delete[] policyWeights0;

		delete[] valueWeights4;
		delete[] valueWeights2;
		delete[] valueWeights1;
	    delete[] valueWeights0;

		delete[] ww4;
		delete[] ww2;
		delete[] ww1;
		delete[] ww0;

		
	}
}


void DataRecord::saveToMat()
{

	tomat::push(drecord.policyWeights4,FEATUREDIM*ACTIONNUM,CHECKPOINTSNUM,"polW4",0,0);
	tomat::push(drecord.policyWeights2,2*FEATUREDIM*ACTIONNUM,CHECKPOINTSNUM,"polW2",0,0);
	tomat::push(drecord.policyWeights1,3*FEATUREDIM*ACTIONNUM,CHECKPOINTSNUM,"polW1",0,0);
	tomat::push(drecord.policyWeights0,6*FEATUREDIM*SCALENUM,CHECKPOINTSNUM,"polW0",0,0);

	tomat::push(drecord.valueWeights4,FEATUREDIM,CHECKPOINTSNUM,"valW4",0,0);
	tomat::push(drecord.valueWeights2,2*FEATUREDIM,CHECKPOINTSNUM,"valW2",0,0);
	tomat::push(drecord.valueWeights1,3*FEATUREDIM,CHECKPOINTSNUM,"valW1",0,0);
	tomat::push(drecord.valueWeights0,6*FEATUREDIM,CHECKPOINTSNUM,"valW0",0,0);
	
	tomat::push(drecord.ww4,FEATUREDIM*ACTIONNUM,CHECKPOINTSNUM,"wW4",0,0);
	tomat::push(drecord.ww2,2*FEATUREDIM*ACTIONNUM,CHECKPOINTSNUM,"wW2",0,0);
	tomat::push(drecord.ww1,3*FEATUREDIM*ACTIONNUM,CHECKPOINTSNUM,"wW1",0,0);
	tomat::push(drecord.ww0,6*FEATUREDIM*SCALENUM,CHECKPOINTSNUM,"wW0",0,0);


	tomat::push(drecord.bases44a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B44a",0,0);
	tomat::push(drecord.bases44a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B44b",0,0);
	tomat::push(drecord.bases24a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B24a",0,0);
	tomat::push(drecord.bases24a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B24b",0,0);
	tomat::push(drecord.bases22a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B22a",0,0);
	tomat::push(drecord.bases22a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B22b",0,0);
	tomat::push(drecord.bases13a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B13a",0,0);
	tomat::push(drecord.bases13a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B13b",0,0);
	tomat::push(drecord.bases12a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B12a",0,0);
	tomat::push(drecord.bases12a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B12b",0,0);
	tomat::push(drecord.bases11a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B11a",0,0);
	tomat::push(drecord.bases11a,BASISDIM*BASESNUM,CHECKPOINTSNUM,"B11b",0,0);


	tomat::push(&drecord.tTrain,1,1,"tTrain",0,0);
	

}

