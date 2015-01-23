#ifndef SELECT_H
#define SELECT_H

#include "tpch.h"
class SelectionApp {
private:
    vector<Tuple>      mCpuOutput;			//Output from the CPU reference code
    cl_kernel         mKernel1;			//Kernel object
    cl_kernel         mKernel2;			//Kernel object
    cl_kernel         mKernel2_1;			//Kernel object
    cl_kernel         mKernel2_2;			//Kernel object
    cl_kernel         mKernel3;			//Kernel object
    cl_mem         mTmpOutputBuffer;		//Buffer for output
    cl_mem         mHistogramBuffer;	//Buffer for histogram
    cl_mem         mTotalsBuffer;		//Buffer for histogram
    size_t             mNumElements;		//Number of elements in input
    size_t             mLocalSize;			//Work-group size
    size_t             mGlobalSize;		//Global work-items
    valType			   mThreshold;

public:

    SelectionApp()
    {}

    ~SelectionApp(); 

    size_t GetNumElements() {
        return mNumElements;
    }
   
    
    void SetSizes(size_t groupSize, size_t numElements, valType threshold);
    //Reset the input and output vectors and existing buffers
    void   ResetBuffers();
    void   FreeDevBuffers();
    int    SetBuffers(bool, BmkParams param);					//create buffers and push data to GPU.
    int    SetKernel(BmkParams param);										//create the kernel and set the arguments.
    size_t RunCPUReference(double&, vector<Tuple>, size_t);	//run the CPU reference.

    int    RunKernel(bool, BmkParams param);														//run the kernel and display results.
    void   Run(bool, BmkParams param);												//start the application.
};
#endif
