#ifndef UNIQUE_H
#define UNIQUE_H

#include "tpch.h"
#define k 0 

// ********************************************************************************************
//  Class:  UniqueApp
//
//  Purpose:
//    Defines the variables required to run the UNIQUE operator. 
//	  A unary operator that removes consecutive duplicates in an attribute.
//	  Example: x = {4, 1, 1, 1, 5, 7, 7, 2}
//	  UNIQUE(x) = {4, 1, 5, 7, 2}
//
//  Programmer:  Ifrah Saeed
//  Creation:    2014
//
// ********************************************************************************************


class UniqueApp
{
private:
    vector<Tuple>      mCpuOutput;			//Output from the CPU reference code
	cl_kernel         mKernel1;			//Kernel object
	cl_kernel         mKernel1_1;			//Kernel object
	cl_kernel         mKernel2;			//Kernel object
	cl_kernel         mKernel2_1;			//Kernel object
	cl_kernel         mKernel2_2;			//Kernel object
	cl_kernel         mKernel3;			//Kernel object
	cl_mem         mTmpOutputBuffer;		//Intermediate Buffer
	cl_mem         mHistogramBuffer;	//Intermediate Buffer
	cl_mem         mTotalsBuffer;		//Intermediate Buffer
	size_t             mNumElements;		//Number of elements in input
	size_t             mLocalSize;			//Work-group size
	size_t             mGlobalSize;			//Total number of work-items

public:
	UniqueApp() {}

	~UniqueApp() {}

    void   ResetBuffers();
    void   FreeDevBuffers();

	void InitializeHost();						//generates data and sets vectors.
	int SetBuffers(BmkParams param);			//creates buffers and pushes data to GPU.
    void SetSizes(size_t groupSize, size_t numElements);
	int SetKernel(BmkParams param);							//creates kernels and sets arguments.
    size_t RunCPUReference(double&, vector<Tuple>, size_t);	//run the CPU reference.
	//runs the CPU reference.
	int RunKernel(BmkParams param);				//runs kernels and displays results.
	void Run(BmkParams param);					//starts the application.
	//void Clean();								//cleans memory
};

#endif
