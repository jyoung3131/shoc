#include "unique.hpp"

void UniqueApp::InitializeHost() {

    /*for (size_t i = 0; i < mNumElements; ++i) {
        Tuple in;

        in.key = GetUrandom(1, mNumElements/2 );
        in.valArray[0] =GetUrandom(1, mNumElements/2 );
        in.valArray[1] =GetUrandom(1, mNumElements/2 );
        in.valArray[2] =GetUrandom(1, mNumElements/2 );
        mInput.push_back(in);
    }*/
}

void UniqueApp::SetSizes(size_t groupSize, size_t numElements)
{
        mLocalSize = groupSize;
        mGlobalSize = numElements;
        mNumElements = numElements;
}

//Reset the input and output vectors and existing cl_mem buffers
void UniqueApp::ResetBuffers()
{

}

void UniqueApp::FreeDevBuffers()
{
	cl_int err;

	err = clReleaseMemObject(mHistogramBuffer);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(mTotalsBuffer);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(mTmpOutputBuffer);
	CL_CHECK_ERROR(err);
}

size_t UniqueApp::RunCPUReference(double &t, vector<Tuple>input, size_t numElements) 
{
	unsigned int output_size = 0;

    Timer* inst;
    int start = inst->Start();

    //Clear the CPU reference vector
    mCpuOutput.clear();
	output_size = 0;
    int uniqCnt = 0;
    
    //This operation pushes back the first unique value
    //for a consecutive string of numbers so the result will
    //start with the first value in the input 
    mCpuOutput.push_back(input[0]);

    for (int j=0; j<mNumElements; j++) {
        if(input[j].tuple.valArray[0] != mCpuOutput[uniqCnt].tuple.valArray[0]) {
            mCpuOutput.push_back(input[j]);
            uniqCnt++;
        }
    }
	
    output_size = mCpuOutput.size();

	t = t + (inst->Stop(start,"CPU"));

	return output_size;
}



int UniqueApp::SetKernel(BmkParams param) {
    cl_int err;

    mKernel1 = clCreateKernel(program, "Unique", &err);
    CL_CHECK_ERROR(err)

    mKernel1_1 = clCreateKernel(program, "CornerCase", &err);
    CL_CHECK_ERROR(err)

    mKernel2 = clCreateKernel(program, "PrefixSum", &err);
    CL_CHECK_ERROR(err)

    mKernel2_1 = clCreateKernel(program, "PrefixSum", &err);
    CL_CHECK_ERROR(err)

    mKernel2_2 = clCreateKernel(program, "Sum", &err);
    CL_CHECK_ERROR(err)

    mKernel3 = clCreateKernel(program, "UniqueGather", &err);
    CL_CHECK_ERROR(err)

    unsigned int groups = (mGlobalSize+mLocalSize-1)/mLocalSize;
    unsigned int subGroups = (groups+mLocalSize-1)/mLocalSize;
    unsigned int isTotal = 0;
    if(subGroups > 1) isTotal = 1;
    unsigned int noTotal = 0;

    err = clSetKernelArg(mKernel1, 0, sizeof(cl_mem), &(param.memInput[0]));
    err += clSetKernelArg(mKernel1, 1, sizeof(cl_mem), &mTmpOutputBuffer);
    err += clSetKernelArg(mKernel1, 2, sizeof(valType)* mLocalSize, NULL);
    err += clSetKernelArg(mKernel1, 3, sizeof(cl_mem), &mHistogramBuffer);
    err += clSetKernelArg(mKernel1, 4, sizeof(unsigned int), &mNumElements);
    CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel1_1, 0, sizeof(cl_mem), &mTmpOutputBuffer);
    err += clSetKernelArg(mKernel1_1, 1, sizeof(cl_mem), &mHistogramBuffer);
    err += clSetKernelArg(mKernel1_1, 2,sizeof(unsigned int), &mLocalSize);
    CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel2, 0,sizeof(cl_mem), &mHistogramBuffer);
    err += clSetKernelArg(mKernel2, 1,sizeof(cl_mem), &mTotalsBuffer);
    err += clSetKernelArg(mKernel2, 2,sizeof(unsigned int),&isTotal);
    CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel2_1, 0,sizeof(cl_mem), &mTotalsBuffer);
    err += clSetKernelArg(mKernel2_1, 1,sizeof(cl_mem), &mTotalsBuffer);
    err += clSetKernelArg(mKernel2_1, 2,sizeof(unsigned int),&noTotal);
    CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel2_2, 0,sizeof(cl_mem), &mHistogramBuffer);
    err += clSetKernelArg(mKernel2_2, 1,sizeof(cl_mem), &mTotalsBuffer);
    CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 0,sizeof(cl_mem),  &(param.memOutput));
    err += clSetKernelArg(mKernel3, 1,sizeof(cl_mem), &mTmpOutputBuffer);
    err += clSetKernelArg(mKernel3, 2,sizeof(cl_mem), &mHistogramBuffer);
    CL_CHECK_ERROR(err);
    return CL_SUCCESS;
}

int UniqueApp::SetBuffers(BmkParams param) {
    cl_int err;
    size_t numWorkGroups = mGlobalSize / mLocalSize;
    size_t dataSizeInBytes = sizeof(valType)* mNumElements;
    size_t subGroups = (numWorkGroups+mLocalSize-1)/mLocalSize;
    double t;
	Event evKrnDataIn("Data Write");

    mTmpOutputBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, dataSizeInBytes, 0, &err);
    CheckError(err, CL_SUCCESS, "Failed to create output buffer.");

    mHistogramBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(numWorkGroups+1), 0, &err);
    CheckError(err, CL_SUCCESS, "Failed to create histogram buffer.");

    mTotalsBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(subGroups+1), 0, &err);
    CheckError(err, CL_SUCCESS, "Failed to create final output buffer.");

    //Pull the value from each tuple and put this in device memory
    vector<valType> inputColumn;
    for (std::vector<Tuple>::iterator it = param.mInputVals[0].begin(); it != param.mInputVals[0].end(); it++) {
        inputColumn.push_back((*it).tuple.valArray[k]);
    }
	
    err = clEnqueueWriteBuffer(*(param.queue), param.memInput[0], CL_TRUE, 0, dataSizeInBytes,
				&inputColumn.front(), 0, NULL, &evKrnDataIn.CLEvent());
	CL_CHECK_ERROR(err);
	err = clWaitForEvents (1, &evKrnDataIn.CLEvent());
	CL_CHECK_ERROR(err);

	evKrnDataIn.FillTimingInfo();
	if (param.verbose) evKrnDataIn.Print(cerr);

	char sizeStr[256];
	unsigned long long dataSizekB = dataSizeInBytes/1024;	
	sprintf(sizeStr, "%7.llukB",dataSizekB);
	double nsToSec = 1.e-9;
	double dataInSec;
	AddTimeResultToDB(param, evKrnDataIn.StartEndRuntime(), dataSizeInBytes, "DataIn"); 


	//(param.resultDB)->AddResult("DataIn", sizeStr,"s", dataInSec);


    return CL_SUCCESS;
}

int UniqueApp::RunKernel(BmkParams param) {
    cl_int err;
    size_t numGroups = mGlobalSize / mLocalSize;
    size_t subGroups = numGroups/mLocalSize;
	size_t dataSizeInBytes = sizeof(valType)* mNumElements;
    
    //vector<valType> gpuResult(mNumElements);
    Event profilingEvt1("Kernel 1");
    Event profilingEvt1_1("Kernel 1_1");
    Event profilingEvt2("Kernel 2");
    Event profilingEvt2_1("Kernel 2_1");
    Event profilingEvt2_2("Kernel 2_2");
    Event profilingEvt3("Kernel 3");
	Event evKrnDataOut("Data Read");
    double cpuTime;
	valType* mHostOutput;
    valType *mapPtr1;
    valType *mapPtr2;
	
    //Currently Kernel 2, prefix sum must always use WG=256
    size_t preSumWgSz = 256;

    double time1Sec=0, time1_1Sec=0, time2Sec=0, time2_1Sec=0, time2_2Sec=0, time3Sec=0, timeDataOutSec = 0;

    double nsToSec = 1.e-9;
	double dataInSec;

    char sizeStr[256];
    unsigned long long dataSizekB = (mNumElements*16)/1024;
    sprintf(sizeStr, "%7.llukB",dataSizekB);

	
    //-------------Kernel 1-------------------------
    
    err = clEnqueueNDRangeKernel(*(param.queue), mKernel1, 1, NULL, &mGlobalSize,
								 &mLocalSize, 0, NULL, &profilingEvt1.CLEvent());
    CL_CHECK_ERROR(err);
    CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");

    err = clWaitForEvents (1, &profilingEvt1.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");
    profilingEvt1.FillTimingInfo();
    if (param.verbose) profilingEvt1.Print(cerr);

    //-------------Kernel 1_1-------------------------
    if(numGroups > 1) {
        size_t t = numGroups - 1;
        err = clEnqueueNDRangeKernel(*(param.queue), mKernel1_1, 1, NULL,
                                     &t, NULL, 0, NULL,
                                     &profilingEvt1_1.CLEvent());
        CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");
        err = clWaitForEvents (1, &profilingEvt1_1.CLEvent());
        CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");
        profilingEvt1_1.FillTimingInfo();
        if (param.verbose) profilingEvt1_1.Print(cerr);
    }

    //size_t lSize = numGroups < mLocalSize ? numGroups : mLocalSize;
    size_t lSize = numGroups < preSumWgSz ? numGroups : preSumWgSz;

    //-------------Kernel 2-------------------------
    err = clEnqueueNDRangeKernel(*(param.queue), mKernel2, 1, NULL,
                                 &numGroups,
                                 &lSize, 0,
                                 NULL,
                                 &profilingEvt2.CLEvent());
    CL_CHECK_ERROR(err);
    CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");
    err = clWaitForEvents (1, &profilingEvt2.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");
    profilingEvt2.FillTimingInfo();
    if (param.verbose) profilingEvt2.Print(cerr);


    //-------------Kernel 2_1-------------------------
    size_t localSize;
    if(subGroups > 1) {
	localSize = subGroups < preSumWgSz ? subGroups : preSumWgSz;
        //size_t localSize = subGroups < mLocalSize ? subGroups : mLocalSize;
        err = clEnqueueNDRangeKernel(*(param.queue), mKernel2_1, 1, NULL,
                                     &subGroups,
                                     &localSize, 0,
                                     NULL,
                                     &profilingEvt2_1.CLEvent());
    CL_CHECK_ERROR(err);
        CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");
        err = clWaitForEvents (1, &profilingEvt2_1.CLEvent());
        CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");
        profilingEvt2_1.FillTimingInfo();
        if (param.verbose) profilingEvt2_1.Print(cerr);

    //-------------Kernel 2_2-------------------------

        err = clEnqueueNDRangeKernel(*(param.queue), mKernel2_2, 1, NULL,
                                     &numGroups,
                                     &mLocalSize, 0,
                                     NULL,
                                     &profilingEvt2_2.CLEvent());
        CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");
        err = clWaitForEvents (1, &profilingEvt2_2.CLEvent());
        CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");
        profilingEvt2_2.FillTimingInfo();
        if (param.verbose) profilingEvt2_2.Print(cerr);
    }
    
    //-------------Kernel 3-------------------------

	printf ("Enqueing kernel3 %lu %lu\n",mGlobalSize, mLocalSize);
    err = clEnqueueNDRangeKernel(*(param.queue), mKernel3, 1, NULL,
                                 &mGlobalSize,
                                 &mLocalSize, 0,
                                 NULL,
                                 &profilingEvt3.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");
    err = clWaitForEvents (1, &profilingEvt3.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");
    profilingEvt3.FillTimingInfo();
    if (param.verbose) profilingEvt3.Print(cerr);

	//----------Results---------------------------------
    
    time1Sec = profilingEvt1.StartEndRuntime()*nsToSec;
    (param.resultDB)->AddResult("SubKernel1", sizeStr,"s", time1Sec);

    if(numGroups > 1) {
        time1_1Sec = profilingEvt1_1.StartEndRuntime()*nsToSec;
        (param.resultDB)->AddResult("SubKernel1_1", sizeStr,"s", time2Sec);
    }

    time2Sec = profilingEvt2.StartEndRuntime()*nsToSec;
    (param.resultDB)->AddResult("SubKernel2", sizeStr,"s", time2Sec);

    if(subGroups > 1) {
        time2_1Sec = profilingEvt2_1.StartEndRuntime()*nsToSec;
        (param.resultDB)->AddResult("SubKernel2_1", sizeStr,"s", time2_1Sec);

        time2_2Sec = profilingEvt2_2.StartEndRuntime()*nsToSec;
        (param.resultDB)->AddResult("SubKernel2_2", sizeStr,"s", time2_2Sec);
    }

    time3Sec = profilingEvt3.StartEndRuntime()*nsToSec;
    (param.resultDB)->AddResult("SubKernel3", sizeStr,"s", time3Sec);

    double krnl_time = time1Sec + time1_1Sec + time2Sec + time2_1Sec + time2_2Sec + time3Sec;
    (param.resultDB)->AddResult("UniqueKernel", sizeStr,"s", krnl_time);

	//----------Transfer Data Out---------------------------------
	err = clEnqueueReadBuffer(*(param.queue), param.memOutput, CL_TRUE, 0, dataSizeInBytes,
				&param.mOutputVals.front(), 0, NULL, &evKrnDataOut.CLEvent());
    CL_CHECK_ERROR(err);
	err = clWaitForEvents (1, &evKrnDataOut.CLEvent());
	CL_CHECK_ERROR(err);

	evKrnDataOut.FillTimingInfo();
	if (param.verbose) evKrnDataOut.Print(cerr);

	dataInSec = evKrnDataOut.StartEndRuntime()*nsToSec; 
	(param.resultDB)->AddResult("DataOut", sizeStr,"s", dataInSec);

    int diffCount = 0;
    for(int i=0; i<mCpuOutput.size(); i++) {
			if (mCpuOutput[i].tuple.valArray[0] != param.mOutputVals[i].tuple.valArray[0]) {
				if(param.verbose)
					printf("%d %u %u\n",i, mCpuOutput[i].tuple.valArray[0], param.mOutputVals[i].tuple.valArray[0]);
            diffCount++;
        }
    }
    if(diffCount == 0)
        Println("Verification outcome : PASSED!");
    else
        Println("Verification outcome : FAILED!");

    err = clReleaseKernel(mKernel1);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(mKernel1_1);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(mKernel2);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(mKernel2_1);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(mKernel2_2);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(mKernel3);
    CL_CHECK_ERROR(err);

    return CL_SUCCESS;
}

void UniqueApp::Run(BmkParams param) {
    CheckError(SetKernel(param), CL_SUCCESS, "setKernel failed.");
    CheckError(RunKernel(param), CL_SUCCESS, "runKernel failed.");
}
