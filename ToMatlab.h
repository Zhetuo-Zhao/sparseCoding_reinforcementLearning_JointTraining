
#pragma once
#include <engine.h>
#include "config.h"


extern Engine *ep;
namespace tomat
{
	void start();
	void close();

	//void push(const double* m,int rowNum,int colNum, char* c);
	template<class T>
	void push(const T* m,int rowNum,int colNum,char* c,bool gpu = 1,bool trans = 0)
	{
		double* store = new double[rowNum*colNum];
		if(gpu)
		{
			T *tmp = new T[rowNum*colNum];
			cudaMemcpy(tmp,m,sizeof(T)*rowNum*colNum,cudaMemcpyDeviceToHost);
			std::copy(tmp,tmp+rowNum*colNum,store);
			delete [] tmp;
		}
		else
			std::copy(m,m+rowNum*colNum,store);	
	
		mxArray *mT;
		if(trans)
		{ 
			mT = mxCreateDoubleMatrix(colNum, rowNum, mxREAL);
			memcpy((void *)mxGetPr(mT), store, mxGetNumberOfElements(mT)*sizeof(double));
			engPutVariable(ep, c, mT);
			
			char str[100];
			sprintf(str,"%s = %s'",c,c);
			engEvalString(ep,str);
		}else
		{
			mT = mxCreateDoubleMatrix(rowNum,colNum, mxREAL);
			memcpy((void *)mxGetPr(mT), store, mxGetNumberOfElements(mT)*sizeof(double));
			engPutVariable(ep, c, mT);
		}
		mxDestroyArray(mT);
	}
	template<class T>
	void get(T* m,char* c,bool gpu = 1,bool trans = 0)
	{
		char str[100];
		mxArray *mT;
		if(trans)
		{
			sprintf(str,"tmpForCppFatch = %s'",c);
			engEvalString(ep,str);
			mT = engGetVariable(ep, "tmpForCppFatch");
		}
		else
			mT = engGetVariable(ep, c);
		int size = mxGetNumberOfElements(mT);
		double* store = new double[size];
		memcpy(store,(void *)mxGetPr(mT), size*sizeof(double));

		if(gpu)
		{
			T *tmp = new T[size];
			std::copy(store,store+size,tmp);
			cudaMemcpy(m,tmp,sizeof(T)*size,cudaMemcpyHostToDevice);
			delete [] tmp;
		}
		else
			std::copy(store,store+size,m);

		mxDestroyArray(mT);
		delete [] store;
		
	}
}
