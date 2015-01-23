#ifndef PRODUCT_H
#define PRODUCT_H

#include "tpch.h"

// ********************************************************************************************
//  Class: ProductApp
//
//  Purpose:
//    Defines the variables required to run the PRODUCT RA operator. 
//	  A binary operator that combines every OldTuple of one input relation to all of the
//	  tuples of the second input relation to
//	  produce a new relation.
//	  Example: x={(4, a), (1, b)}, y={(1, c)}
//	  PRODUCT(x,y) -> {(4, a, 1, c), (1, b, 1, c)}
//
//  Programmer:  Ifrah Saeed, Jeffrey Young
//  Creation:    2014
//
// ********************************************************************************************


class ProductApp
{
private:
	vector<Tuple>		        mCpuOutput;			//Output vector
	cl_kernel					mKernel1;			//Kernel object

	size_t						mNumLeftElements;	//Number of elements in left input
	size_t						mNumRightElements;	//Number of elements in right input
	size_t						mLocalSize;			//Work-group size
	size_t						mGlobalSize;		//Total number of work-items

public:
	ProductApp()
	{}

	~ProductApp()
	{}


    void SetSizes(size_t groupSize, size_t numLeftElements, size_t numRightElements);
    
    void   ResetBuffers();
    void   FreeDevBuffers();

    int    SetBuffers(BmkParams param);					//create buffers and push data to GPU.
    int    SetKernel(BmkParams param);										//create the kernel and set the arguments.
    size_t RunCPUReference(double&, vector<Tuple>, vector<Tuple>, size_t, size_t);	//run the CPU reference.
	int RunKernel(BmkParams param);					//runs kernels and displays results.
	void Run(BmkParams param);							//starts the application.
};
#endif
