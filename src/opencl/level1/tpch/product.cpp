#include "product.hpp"

//Reset the input and output vectors and existing cl_mem buffers
//Currently no temp buffers for this benchmark
void ProductApp::ResetBuffers()
{
}

//Currently no local device buffers for this benchmark
void ProductApp::FreeDevBuffers()
{
}


void ProductApp::SetSizes(size_t groupSize, size_t numLeftElements, size_t numRightElements)
{
        mLocalSize = groupSize;
        mGlobalSize = numLeftElements;
        mNumLeftElements = numLeftElements;
        mNumRightElements = numRightElements;
}

size_t ProductApp::RunCPUReference(double &t, vector<Tuple> input1, vector<Tuple> input2, size_t mNumLeftElements, size_t mNumRightElements)
{

	Timer* inst;
	unsigned int output_size = 0;
	int start = inst->Start();
	mCpuOutput.clear();

	for(int l=0;l<mNumLeftElements;l++){
		for(int r=0;r<mNumRightElements;r++){
			Tuple tpl;
			tpl.prod_tuple.keyArray[0]	= input1[l].tuple.key;
            		tpl.prod_tuple.valArray[0] 	= input1[l].tuple.valArray[0];
			tpl.prod_tuple.keyArray[1]	= input2[r].tuple.key;
            		tpl.prod_tuple.valArray[1] 	= input2[r].tuple.valArray[0];
			mCpuOutput.push_back(tpl);
		}
	}
	output_size = mCpuOutput.size();
	
    t = inst->Stop(start,"CPU");

    return output_size;

	return 0;
}

int ProductApp::SetKernel(BmkParams param)
{
	cl_int err;
		
	mKernel1 = clCreateKernel(program, "Product", &err);
    CL_CHECK_ERROR(err)
	unsigned int groups = (mGlobalSize+mLocalSize-1)/mLocalSize;
	unsigned int subGroups = (groups+mLocalSize-1)/mLocalSize;
	unsigned int isTotal = 0;
	if(subGroups > 1) isTotal = 1; 
	unsigned int noTotal = 0;

	err = clSetKernelArg(mKernel1, 0, sizeof(cl_mem), &param.memInput[0]);
	err += clSetKernelArg(mKernel1, 1, sizeof(cl_mem),&param.memInput[1]);
	err += clSetKernelArg(mKernel1, 2, sizeof(cl_mem),&param.memOutput);
	err += clSetKernelArg(mKernel1, 3, /*sizeof(Tuple)* mLocalSize*/param.devBufferSz, NULL);
	err += clSetKernelArg(mKernel1, 4, sizeof(unsigned int), &mNumLeftElements);
	err += clSetKernelArg(mKernel1, 5, sizeof(unsigned int), &mNumRightElements);
    CL_CHECK_ERROR(err);
	return CL_SUCCESS;
}

int ProductApp::SetBuffers(BmkParams param)
{
	cl_int err;
	size_t numWorkGroups = mGlobalSize / mLocalSize;
	size_t lDataSizeInBytes = sizeof(Tuple)* mNumLeftElements;
	size_t rDataSizeInBytes = sizeof(Tuple)* mNumRightElements;
	size_t subGroups = (numWorkGroups+mLocalSize-1)/mLocalSize;
	Event evKrnDataIn1("Data1 Write");
	Event evKrnDataIn2("Data2 Write");
	
	/*mLeftBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_ONLY, lDataSizeInBytes,0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create input buffer.");

	mRightBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_ONLY, rDataSizeInBytes, 0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create input buffer.");
	param.mOutputValsBuffer = clCreateBuffer(*(param.ctx), CL_MEM_WRITE_ONLY, mNumLeftElements*mNumRightElements*sizeof(Tuple), 0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create output buffer.");
*/
	err = clEnqueueWriteBuffer(*(param.queue), param.memInput[0], CL_TRUE, 0, lDataSizeInBytes,
				&(param.mInputVals[0].front()), 0, NULL, &evKrnDataIn1.CLEvent());
	CL_CHECK_ERROR(err);
	
    err = clWaitForEvents (1, &evKrnDataIn1.CLEvent());
	CL_CHECK_ERROR(err);
	
    evKrnDataIn1.FillTimingInfo();
	if (param.verbose) evKrnDataIn1.Print(cerr);
    
	err = clEnqueueWriteBuffer(*(param.queue), param.memInput[1], CL_TRUE, 0, rDataSizeInBytes,
				&(param.mInputVals[1].front()), 0, NULL, &evKrnDataIn2.CLEvent());
	CL_CHECK_ERROR(err);
	
    err = clWaitForEvents (1, &evKrnDataIn2.CLEvent());
	CL_CHECK_ERROR(err);
	
    evKrnDataIn2.FillTimingInfo();
	if (param.verbose) evKrnDataIn2.Print(cerr);

	char sizeStr[256];
	unsigned long long dataSizekB = (lDataSizeInBytes + rDataSizeInBytes)/1024;	
	sprintf(sizeStr, "% 7llukB",dataSizekB);
	double nsToSec = 1.e-9;
	double dataInSec;
	dataInSec = evKrnDataIn1.StartEndRuntime()*nsToSec; 
	dataInSec += evKrnDataIn2.StartEndRuntime()*nsToSec; 
	
    //Add time for data transfer to the database
    (param.resultDB)->AddResult("DataIn", sizeStr,"s", dataInSec);
	
	return CL_SUCCESS;
}

int ProductApp::RunKernel(BmkParams param)
{
	cl_int err;
	size_t numGroups = mGlobalSize / mLocalSize;
	size_t subGroups = numGroups/mLocalSize;
	
	Event profilingEvt1("Kernel");
	Event profilingEvt2("DataOut");
	double time1Sec = 0, dataOutSec = 0;
	double cpuTime;
	size_t dataSizeInBytes = sizeof(Tuple)* mNumLeftElements *mNumRightElements;
	//ProductTuple *mCpuOutput;
	
	//-------------Kernel 1-------------------------
//printf(" %lu %lu\n", mGlobalSize, mLocalSize); 
    err = clEnqueueNDRangeKernel(*(param.queue), mKernel1, 1, NULL, &mGlobalSize,
								 &mLocalSize, 0, NULL, &profilingEvt1.CLEvent());
    CL_CHECK_ERROR(err);
    err = clWaitForEvents (1, &profilingEvt1.CLEvent());
	CheckError(err, CL_SUCCESS, "Failed to finish command queue commands.");

    profilingEvt1.FillTimingInfo();
    if (param.verbose) profilingEvt1.Print(cerr);
	
	//----------Results---------------------------------
	
    double nsToSec = 1.e-9;
	char sizeStr[256];
	unsigned long long dataSizekB = (dataSizeInBytes)/1024;	
	//Filler until we handle multiple sizes 
	sprintf(sizeStr, "% 7llukB",dataSizekB);
    time1Sec = profilingEvt1.StartEndRuntime()*nsToSec;
    (param.resultDB)->AddResult("ProductKernel", sizeStr,"s", time1Sec);

	//----------Data Transfer Out-------------------------------
	
	err = clEnqueueReadBuffer(*(param.queue), param.memOutput, CL_TRUE, 0,
			 dataSizeInBytes,
				&param.mOutputVals.front(), 0, NULL, &profilingEvt2.CLEvent());
	CL_CHECK_ERROR(err);

	err = clWaitForEvents (1, &profilingEvt2.CLEvent());
	CL_CHECK_ERROR(err);

	profilingEvt2.FillTimingInfo();
	if (param.verbose) profilingEvt2.Print(cerr);
	dataOutSec = profilingEvt2.StartEndRuntime()*nsToSec; 
	(param.resultDB)->AddResult("DataOut", sizeStr,"s", dataOutSec);

	//----------Validate results against CPU-----------------------
	//RunCPUReference(cpuTime);
if (param.verbose) {
for (int i = 0; i < param.numElems; i++) {
	printf("%u %u                 ", param.mInputVals[0][i].tuple.key, param.mInputVals[0][i].tuple.valArray[0]);
	printf("%u %u\n", param.mInputVals[1][i].tuple.key, param.mInputVals[1][i].tuple.valArray[0]);
}
}

//printf("first element of result is %u %u\n", param.mOutputVals[0].prod_tuple.keyArray[0], param.mOutputVals[0].prod_tuple.valArray[0]);	
	int diffCount = 0;
	for(int i=0;i<mCpuOutput.size();i++){
		if (       param.mOutputVals[i].prod_tuple.keyArray[0] != mCpuOutput[i].prod_tuple.keyArray[0] 
			|| param.mOutputVals[i].prod_tuple.valArray[0] != mCpuOutput[i].prod_tuple.valArray[0] 
			|| param.mOutputVals[i].prod_tuple.keyArray[1] != mCpuOutput[i].prod_tuple.keyArray[1]
			|| param.mOutputVals[i].prod_tuple.valArray[1] != mCpuOutput[i].prod_tuple.valArray[1]){

			if(param.verbose) {
				printf("%d %u %u %u %u\n  %u %u %u %u\n",i, 
					param.mOutputVals[i].prod_tuple.keyArray[0], 
					param.mOutputVals[i].prod_tuple.valArray[0],
					param.mOutputVals[i].prod_tuple.keyArray[1],
					param.mOutputVals[i].prod_tuple.valArray[1],
					mCpuOutput[i].prod_tuple.keyArray[0],
					mCpuOutput[i].prod_tuple.valArray[0], 	
					mCpuOutput[i].prod_tuple.keyArray[1],
					mCpuOutput[i].prod_tuple.valArray[1]);
				cin.get();
			}
			diffCount++;
		}
	}	
	
	if(diffCount == 0)
			Println(" Verification outcome : PASSED!"); 
	else
			Println(" Verification outcome : FAILED!"); 

    //Release the kernel
	err = clReleaseKernel(mKernel1);
	CL_CHECK_ERROR(err);
	
    return CL_SUCCESS;
}

void ProductApp::Run(BmkParams param)
{
	CheckError(SetKernel(param), CL_SUCCESS, "setKernel failed.");
	CheckError(RunKernel(param), CL_SUCCESS, "runKernel failed.");
}
