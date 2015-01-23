#include "add.hpp"

void AddApp::InitializeHost()
{
	srand((unsigned)time(0));
	for(int sz=0; sz<mNumElements; sz++)
	{
		mInput1[sz] = (rand()%10 + 1) + (float)((rand() << 15 + rand()) & ((1 << 24) - 1)) / (1 << 24);
				
	}
	mInput2 = 1;
}

void AddApp::RunCPUReference(double &t)
{
	//Timer timer;
	//timer.reset();
	unsigned int sz;

	Timer *inst;
	int start = inst->Start();
	//timer.start();
	for(int iter=0;iter<ITER_CPU;iter++){

		for(unsigned int i=0;i<mNumElements;i++){
			mOutput[i] = mInput1[i] + mInput2;
		}
	}
	//timer.stop();
	t = t + (inst->Stop(start, "CPU"))/ITER_CPU;
	//time elapsed
	//t = t + (timer.getElapsedTime()/ITER_CPU);
}

int AddApp::SetKernel()
{
	cl_int err;

	//create kernel
	try
	{
		mKernel1 = cl::Kernel(program, "Addition", &err);
	}
	catch (cl::Error e)
	{
		Println("Error creating kernel (code - %d) : %s.", e.err(), e.what());
		return EXIT_FAILURE;
	}


	unsigned int groups = (mGlobalSize+mLocalSize-1)/mLocalSize;
	

	//set kernel arguments
	try
	{
		err = mKernel1.setArg(0, mInput1Buffer);
		err += mKernel1.setArg(1, sizeof(unsigned int), &mInput2);
		err += mKernel1.setArg(2, mOutputBuffer);
		err += mKernel1.setArg(3, sizeof(unsigned int), &mNumElements);
	}
	catch (cl::Error e)
	{
		Println("Error setting kernel args (code - %d) : %s.", e.err(), e.what());
		return EXIT_FAILURE;
	}

	return CL_SUCCESS;
}

int AddApp::SetBuffers()
{
	cl_int err;
	size_t numWorkGroups = mGlobalSize / mLocalSize;
	size_t dataSizeInBytes = sizeof(float)* mNumElements;
	size_t subGroups = (numWorkGroups+mLocalSize-1)/mLocalSize;
	
	Timer* inst;
	//Timer timer;
	//timer.reset();
	double t;

	//create input buffer
	mInput1Buffer = cl::Buffer(context, CL_MEM_READ_ONLY, dataSizeInBytes, 0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create input buffer.");


	//create output buffer
	mOutputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, dataSizeInBytes, 0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create output buffer.");


	//timer.start();
	int start = inst->Start();
	for(int i=0;i<ITER_GPU;i++){
	//push data to GPU
		err = commandQueue.enqueueWriteBuffer(mInput1Buffer, CL_TRUE, 0, dataSizeInBytes,
											&mInput1.front(), 0, 0);
		CheckError(err, CL_SUCCESS, "Failed to write input buffer.");
	}
	
	//timer.stop();
	t = inst->Stop(start, "Writing input buffer")/ITER_GPU;
	//t = timer.getElapsedTime()/ITER_GPU;

	Println("Writing buffer: [%f secs]", t);

	return CL_SUCCESS;
}

int AddApp::RunKernel()
{
	cl_int err;
	size_t numGroups = mGlobalSize / mLocalSize;
	size_t subGroups = numGroups/mLocalSize;
	valType gpuResult = 0;
	cl::Event profilingEvt1;
	double cpuTime;
	float *mapPtr;

	cl_ulong startTime1, endTime1;
	cl_ulong time1 = 0;
	

	Timer* inst;
	//Timer timer;
	//timer.reset();
	//timer.start();
	int start = inst->Start();
	for(int i=0;i<ITER_GPU;i++){
		//execute the kernel.
		err = commandQueue.enqueueNDRangeKernel(mKernel1, cl::NullRange,
											  cl::NDRange(mGlobalSize),
											  cl::NDRange(mLocalSize),
											  NULL,
											  &profilingEvt1);
		CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");

		err = commandQueue.finish();
		CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");

		profilingEvt1.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime1);
		profilingEvt1.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime1);
		time1 += endTime1 - startTime1;


	}
	//timer.stop();

	double time = inst->Stop(start, "Kernel execution")/ITER_GPU;
	//time = timer.getElapsedTime()/ITER_GPU;
	
	start = inst->Start();
	//timer.start();
	//for(int i=0;i<ITER_GPU;i++){
	for(int i=0;i<10;i++){
		mapPtr = (float*)commandQueue.enqueueMapBuffer(mOutputBuffer, CL_TRUE, CL_MAP_READ, 0,
			sizeof(float)* mNumElements, 0, 0, &err);
		CheckError(err, CL_SUCCESS, "Failed to map output buffer.");
	}
	//timer.stop();
	//double writing = inst->Stop(start, "Writing output buffer")/ITER_GPU;
	double writing = inst->Stop(start, "Writing output buffer")/10;
	//writing = timer.getElapsedTime()/ITER_GPU;

	Println("writing output buffer:%f",writing);

	
	Println();
	
	//runCPUReference(cpuTime);

	//double time = timer.getElapsedTime()/ITER_GPU;

	Println("Total elements    : %d", mNumElements);
	Println("Total work-items  : %d", mGlobalSize);
	Println("Work-group size   : %d", mLocalSize);
	Println("Total work-groups : %d", numGroups);
	Println();
	
	//check if the CPU and GPU results match.
	int diffCount = 0;
	for(int i=0;i<mOutput.size();i++){
		if (mOutput[i] != mapPtr[i]){
			printf("%d %f %f\t",i, mOutput[i], mapPtr[i]);
			diffCount++;
		}
	}
	if(diffCount == 0)
			Println("Verification outcome : PASSED!"); 
	else
			Println("Verification outcome : FAILED!"); 

	//unmap the pointer to output buffer.
	err = commandQueue.enqueueUnmapMemObject(mOutputBuffer, mapPtr, 0, 0);
	CheckError(err, CL_SUCCESS, "Failed to unmap output buffer.");

	Println();
	//Println("CPU : [%f secs]", cpuTime);
	Println("GPU : kernel_sum:[%f secs] \nTotal:[%f secs]", (time1/ITER_GPU)*1.e-9, time);
	
	Println();

	return CL_SUCCESS;
}

void AddApp::Run()
{
	CheckError(SetKernel(), CL_SUCCESS, "setKernel failed.");
	CheckError(RunKernel(), CL_SUCCESS, "runKernel failed.");
}
