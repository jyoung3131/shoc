#include "tpch.h"

class ProjectApp
{
	private:
		vector<valType>      mCpuOutput;			//Output from the CPU reference code
		cl_kernel        mKernel1;			//Kernel object

		cl_mem         mTmpOutputBuffer;		//Buffer for output
		size_t             mNumElements;		//Number of elements in input
		size_t             mLocalSize;			//Work-group size
		size_t             mGlobalSize;		//Global work-items

	public:

		ProjectApp()
		{}

		~ProjectApp()
		{}

		void SetSizes(size_t groupSize, size_t numElements);
		//Reset the input and output vectors and existing buffers
		void   ResetBuffers();
		void   FreeDevBuffers();
		int    SetBuffers(BmkParams param);					//create buffers and push data to GPU.
		int    SetKernel(BmkParams param);										//create the kernel and set the arguments.
		size_t RunCPUReference(double&, vector<Tuple>, size_t, size_t);	//run the CPU reference.

		int    RunKernel(BmkParams param);														//run the kernel and display results.
		void   Run(BmkParams param);												//start the application.
};
