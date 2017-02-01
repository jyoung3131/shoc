#ifndef INNER_JOIN_H
#define INNER_JOIN_H 

#include "tpch.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

class InnerJoinApp
{
	private:
		vector<Tuple>      mLeft;				//input elements
		vector<Tuple>      mRight;				//input elements
		vector<Tuple>	   mOutput;				//output
		vector<Tuple>	   mCpuOutput;			//Output vector
		cl_kernel         mKernel1;			//Kernel object
		cl_kernel         mKernel2;			//Kernel object
		cl_kernel         mKernel2_1;			//Kernel object
		cl_kernel         mKernel2_2;			//Kernel object
		cl_kernel         mKernel3;			//Kernel object
		cl_kernel         mKernel4;			//Kernel object
		cl_kernel         mKernel4_1;			//Kernel object
		cl_kernel         mKernel4_2;			//Kernel object
		cl_kernel         mKernel5;			//Kernel object
		cl_mem         mOutputBuffer;		//Buffer for output
		cl_mem         mHistogramBuffer;	//Buffer for histogram
		cl_mem         mTotalsBuffer;		//Buffer 
		cl_mem         mLowerBuffer;		//Buffer for lower bounds
		cl_mem         mUpperBuffer;		//Buffer for upper bounds
		cl_mem         mOutBoundsBuffer;	//Buffer 
		//cl_mem         mFinalOutBuffer;		//Buffer for final output
		size_t             mNumLeftElements;	//Number of elements in input
		size_t             mNumRightElements;	//Number of elements in input
		size_t             mLocalSize;			//Work-group size
		size_t             mGlobalSize;			//Global work-items
		bool           isFirst;             //Is this the first iteration of a join operation

	public:
		InnerJoinApp()
		{}

		~InnerJoinApp()
		{}

		void SetSizes(size_t groupSize, size_t numLeftElements, size_t numRightElements, bool isFirstFlag);

		void   ResetBuffers();
		void   FreeDevBuffers();

		int    SetBuffers(BmkParams param);					//create buffers and push data to GPU.
		int    SetKernel(BmkParams param);										//create the kernel and set the arguments.
		size_t RunCPUReference(double&, vector<Tuple>, vector<Tuple>, size_t, size_t, bool);	//run the CPU reference.
		int RunKernel(BmkParams param);					//runs kernels and displays results.
		void Run(BmkParams param);							//starts the application.

};


#endif
