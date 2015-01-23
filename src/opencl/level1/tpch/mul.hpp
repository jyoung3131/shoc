#include "../tpch.h"

class MulApp
{
private:
	vector<float>	   mInput1;			//input elements
	vector<float>	   mInput2;
	vector<float>      mOutput;			//outputs. One value from each work-group
	cl::Kernel         mKernel1;			//Kernel object
	cl::Buffer         mInput1Buffer;		//Buffer for input
	cl::Buffer         mInput2Buffer;		//Buffer for input
	cl::Buffer         mOutputBuffer;		//Buffer for output
	size_t             mNumElements;		//Number of elements in input
	size_t             mLocalSize;			//Work-group size
	size_t             mGlobalSize;		//Global work-items

public:
	MulApp(size_t groupSize = LOCAL_SIZE, size_t numElements = NUM_ELEMENTS)
		: mLocalSize(groupSize),
		mGlobalSize(numElements),
		mNumElements(numElements)
	{
		mInput1.resize(mNumElements);
		mInput2.resize(mNumElements);
		mOutput.resize(mNumElements);
	}

	~MulApp()
	{}

	cl::Buffer GetOutBuffer(){
		return mOutputBuffer;
	}
	vector<float> GetOutput(){
		return mOutput;
	}
	vector<float> GetInput2(){
		return mInput2;
	}
	

	void InitializeInput1Host();									//generates data and sets vectors.
	void InitializeInput2Host();
	int  SetBuffers(bool, bool, cl::Buffer,cl::Buffer);				//create buffers and push data to GPU.
	int  SetKernel();												//create the kernel and set the arguments.
	void RunCPUReference(double&, vector<float>, vector<float>);	//run the CPU reference.
	int  RunKernel();												//run the kernel and display results.
	void Run();														//start the application.
};
