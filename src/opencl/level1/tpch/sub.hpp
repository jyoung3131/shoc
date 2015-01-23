#include "../tpch.h"

class SubApp
{
private:
	vector<float>	   mInput1;			//input elements
	unsigned int	   mInput2;
	vector<float>	   mOutput;			//outputs. One value from each work-group
	cl::Kernel         mKernel1;			//Kernel object
	cl::Buffer         mInput1Buffer;		//Buffer for input
	cl::Buffer         mInput2Buffer;		//Buffer for input
	cl::Buffer         mOutputBuffer;		//Buffer for output
	size_t             mNumElements;		//Number of elements in input
	size_t             mLocalSize;			//Work-group size
	size_t             mGlobalSize;		//Global work-items

public:
	SubApp(size_t groupSize = LOCAL_SIZE, size_t numElements = NUM_ELEMENTS)
		: mLocalSize(groupSize),
		mGlobalSize(numElements/4),
		mNumElements(numElements)
	{
		mInput1.resize(mNumElements);
		mOutput.resize(mNumElements);
	}

	~SubApp()
	{}

	cl::Buffer GetOutBuffer(){
		return mOutputBuffer;
	}
	vector<float> GetOutput(){
		return mOutput;
	}

	void InitializeHost();				//generates data and sets vectors.
	int  SetBuffers();					//create buffers and push data to GPU.
	int  SetKernel();					//create the kernel and set the arguments.
	void RunCPUReference(double&);		//run the CPU reference.
	int  RunKernel();					//run the kernel and display results.
	void Run();							//start the application.
};
