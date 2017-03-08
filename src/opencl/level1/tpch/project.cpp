#include "project.hpp"

void ProjectApp::SetSizes(size_t groupSize, size_t numElements)
{
        mLocalSize = groupSize;
        mGlobalSize = numElements;
        mNumElements = numElements;
}

//Reset the input and output vectors and existing cl_mem buffers
void ProjectApp::ResetBuffers()
{

}

void ProjectApp::FreeDevBuffers()
{

}

size_t ProjectApp::RunCPUReference(double &t, vector<Tuple> input, size_t numElements, size_t col)
{
	unsigned int output_size = 0;

	Timer* inst;

	int start = inst->Start();
	
    //Clear the CPU reference vector
    mCpuOutput.clear();
	output_size = 0;
		for (unsigned int i = 0; i < numElements; i++) {
				//Tuple tup;
				//tup.tuple.key = input[i].tuple.key;
				//tup.tuple.valArray[0] = input[i].tuple.valArray[0];
				Tuple tup = input[i];
				mCpuOutput.push_back(col ? tup.tuple.valArray[0] : tup.tuple.key);
		}
		output_size = mCpuOutput.size();

	t = t + (inst->Stop(start,"CPU"));

	return output_size;
}

int ProjectApp::SetKernel(BmkParams param)
{
	cl_int err;

	//create kernel
		mKernel1 = clCreateKernel(program, "Project", &err);
    CL_CHECK_ERROR(err);

	unsigned int groups = (mGlobalSize+mLocalSize-1)/mLocalSize;
	
	    err = clSetKernelArg(mKernel1,0, sizeof(cl_mem), &(param.memInput[0]));
		err += clSetKernelArg(mKernel1, 1,sizeof(cl_mem), &(param.memOutput));
		err += clSetKernelArg(mKernel1, 2, sizeof(unsigned int), &mNumElements);
		err += clSetKernelArg(mKernel1, 3, sizeof(unsigned int), &(param.project_col));
    	CL_CHECK_ERROR(err);

	return CL_SUCCESS;
}

int ProjectApp::SetBuffers(BmkParams param)
{
	cl_int err;
	size_t numWorkGroups = mGlobalSize / mLocalSize;
	size_t inputDataSizeInBytes = sizeof(Tuple)* mNumElements;
	size_t subGroups = (numWorkGroups+mLocalSize-1)/mLocalSize;
	Event evKrnDataIn("Data Write");

	//push data to GPU
	err = clEnqueueWriteBuffer(*param.queue, param.memInput[0], CL_TRUE, 0, inputDataSizeInBytes,
											&(param.mInputVals[0].front()), 0, NULL, &evKrnDataIn.CLEvent());
	CL_CHECK_ERROR(err);
	
    err = clWaitForEvents (1, &evKrnDataIn.CLEvent());
	CL_CHECK_ERROR(err);

	evKrnDataIn.FillTimingInfo();
	if (param.verbose) evKrnDataIn.Print(cerr);

	AddTimeResultToDB(param, evKrnDataIn.StartEndRuntime(), inputDataSizeInBytes, "DataIn"); 
	
    //Add Data Write Result to DB

	return CL_SUCCESS;
}

int ProjectApp::RunKernel(BmkParams param)
{
	cl_int err;
	size_t numGroups = mGlobalSize / mLocalSize;
	size_t subGroups = numGroups/mLocalSize;
	Event evKrn1("Kernel 1");
	Event evKrnDataOut("Data Read");
	double cpuTime;
	//valType *mapPtr;
	size_t dataSizeInBytes = sizeof(valType)* mNumElements;

	double time1Sec;
	cl_ulong time1=0;

    //-------------Kernel 1-------------------------
		//execute the kernel.
		err = clEnqueueNDRangeKernel(*(param.queue), mKernel1, 1, NULL,
											  &(mGlobalSize),
											  &(mLocalSize), 0,
											   NULL,
											  &evKrn1.CLEvent());
    
    CL_CHECK_ERROR(err);

    err = clWaitForEvents (1, &evKrn1.CLEvent());
	evKrn1.FillTimingInfo();
	if (param.verbose) evKrn1.Print(cerr);

	
	//----------Kernel Timing Results------------------------------
    AddTimeResultToDB(param, evKrn1.StartEndRuntime(), dataSizeInBytes, "ProjectKernel");
	
    //----------Transfer Data Out---------------------------

    //TODO: Currently just the output value (a set of keys) is transferred out instead
    //of a tuple; 

	vector<valType> mOutput;
	mOutput.resize(param.numElems);
	//err = clEnqueueReadBuffer(*(param.queue), param.memOutput, CL_TRUE, 0, dataSizeInBytes, &(param.mOutputVals.front()), 0, NULL, &evKrnDataOut.CLEvent());

	err = clEnqueueReadBuffer(*(param.queue), param.memOutput, CL_TRUE, 0, dataSizeInBytes, &(mOutput.front()), 0, NULL, &evKrnDataOut.CLEvent());
	CL_CHECK_ERROR(err);

	err = clWaitForEvents (1, &evKrnDataOut.CLEvent());
	CL_CHECK_ERROR(err);
	
    evKrnDataOut.FillTimingInfo();
	if (param.verbose) evKrnDataOut.Print(cerr);

    AddTimeResultToDB(param, evKrnDataOut.StartEndRuntime(), dataSizeInBytes, "Data Out");
    
    //---------Validation------------------------------------	
	//check if the CPU and GPU results match.

if (param.verbose) {
for (int i = 0; i < param.numElems; i++) {
	printf("%u %u                  ", param.mInputVals[0][i].tuple.key, param.mInputVals[0][i].tuple.valArray[0]);
	//if (i < param.mOutputVals.size()) printf("%u %u", mCpuOutput[i], mCpuOutput[i]);
printf("\n");
}
}

	int diffCount = 0;
	for(int i=0;i<mCpuOutput.size();i++){
            if(param.verbose)
//Trouble What we do about param.mOutputVals
			printf("%d %u %u at address %p\n",i, mCpuOutput[i], mOutput[i], &mOutput[i]);

		if (mCpuOutput[i] != mOutput[i]){
            if(param.verbose)
			printf("%d %u %u\n",i, mCpuOutput[i], mOutput[i]);
			diffCount++;
		}
	}

	if(diffCount == 0)
			Println(" Verification outcome : PASSED!"); 
	else
			Println(" Verification outcome : FAILED!"); 


    err = clReleaseKernel(mKernel1);
    CL_CHECK_ERROR(err);
	
    return CL_SUCCESS;
}

void ProjectApp::Run(BmkParams param)
{
	CheckError(SetKernel(param), CL_SUCCESS, "setKernel failed.");
	CheckError(RunKernel(param), CL_SUCCESS, "runKernel failed.");
}
