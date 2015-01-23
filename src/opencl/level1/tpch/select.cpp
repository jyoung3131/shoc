#include "select.hpp"
#include <ctime>
#define ITER_CPU 1
#define ITER_GPU 1

//Destructor
SelectionApp::~SelectionApp() 
{
	//FreeDevBuffers();
}

void SelectionApp::SetSizes(size_t groupSize, size_t numElements, valType threshold)
{
        mLocalSize = groupSize;
        mGlobalSize = numElements;
        mNumElements = numElements;
        mThreshold = threshold;
}

//Reset the input and output vectors and existing cl_mem buffers
void SelectionApp::ResetBuffers()
{

}

void SelectionApp::FreeDevBuffers()
{
	cl_int err;

	err = clReleaseMemObject(mHistogramBuffer);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(mTotalsBuffer);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(mTmpOutputBuffer);
	CL_CHECK_ERROR(err);
}

size_t SelectionApp::RunCPUReference(double &t, vector<Tuple>input, size_t numElements) {
	
	unsigned int output_size = 0;

	Timer* inst;

	int start = inst->Start();
	
    //Clear the CPU reference vector
    mCpuOutput.clear();
	output_size = 0;
		for (unsigned int i = 0; i < numElements; i++) {
			if (input[i].key < mThreshold ) {
				Tuple tuple;
				tuple.key = input[i].key;
				tuple.valArray[0] = input[i].valArray[0];
				mCpuOutput.push_back(tuple);
			}
		}
		output_size = mCpuOutput.size();

	t = t + (inst->Stop(start,"CPU"));

	return output_size;
}

int SelectionApp::SetKernel(BmkParams param) {
	cl_int err;

	//create kernel
	mKernel1 = clCreateKernel(program, "Selection", &err);
	CL_CHECK_ERROR(err);
	
	mKernel2 = clCreateKernel(program, "PrefixSum", &err);
	//mKernel2 = clCreateKernel(program, "PrefixSum2", &err);
	CL_CHECK_ERROR(err);

	//mKernel2_1 = clCreateKernel(program, "PrefixSum2", &err);
	mKernel2_1 = clCreateKernel(program, "PrefixSum", &err);
	CL_CHECK_ERROR(err);

	mKernel2_2 = clCreateKernel(program, "Sum", &err);
	CL_CHECK_ERROR(err);


	mKernel3 = clCreateKernel(program, "Gather", &err);
	CL_CHECK_ERROR(err);

	unsigned int groups = (mGlobalSize+mLocalSize-1)/mLocalSize;
	unsigned int subGroups = (groups+mLocalSize-1)/mLocalSize;
	unsigned int isTotal = 0;
	if(subGroups > 1) isTotal = 1;
	unsigned int noTotal = 0;
	//set kernel arguments
	err = clSetKernelArg(mKernel1,0, sizeof(cl_mem), &(param.memInput[0]));
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel1,1, sizeof(cl_mem), &mTmpOutputBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel1,2,sizeof(unsigned int),&mNumElements);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel1,3, sizeof(cl_mem), &mHistogramBuffer);
	CL_CHECK_ERROR(err);
	//This argument is a local buffer for the Selection kernel
	err = clSetKernelArg(mKernel1,4, sizeof(Tuple)* mLocalSize, NULL);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel1,5, sizeof(valType), &mThreshold);
	CL_CHECK_ERROR(err);

	/*
    //TODO - add the use of PrefixSum2
    uint bin_size = sizeof(valType);
	uint* temp;
	err = clSetKernelArg(mKernel2,0,sizeof(cl_mem), &mHistogramBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2,1,sizeof(cl_mem), &mTotalsBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2,2,sizeof(uint), &bin_size);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2,3,sizeof(uint)*mLocalSize, temp);
	CL_CHECK_ERROR(err);*/
    
	err = clSetKernelArg(mKernel2,0,sizeof(cl_mem), &mHistogramBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2,1,sizeof(cl_mem), &mTotalsBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2,2,sizeof(uint), &isTotal);
	CL_CHECK_ERROR(err);

	err = clSetKernelArg(mKernel2_1,0,sizeof(cl_mem), &mTotalsBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2_1,1,sizeof(cl_mem), &mTotalsBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2_1,2,sizeof(unsigned int),&noTotal);
	CL_CHECK_ERROR(err);

	err = clSetKernelArg(mKernel2_2,0,sizeof(cl_mem), &mHistogramBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel2_2,1,sizeof(cl_mem), &mTotalsBuffer);
	CL_CHECK_ERROR(err);

	err = clSetKernelArg(mKernel3,0,sizeof(cl_mem), &(param.memOutput));
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel3,1,sizeof(cl_mem), &mTmpOutputBuffer);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel3,2,sizeof(unsigned int),&mNumElements);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(mKernel3,3,sizeof(cl_mem), &mHistogramBuffer);
	CL_CHECK_ERROR(err);

	return CL_SUCCESS;
}

int SelectionApp::SetBuffers(bool isFirst, BmkParams param) {
	cl_int err;

	size_t numWorkGroups = mGlobalSize / mLocalSize;
	size_t dataSizeInBytes = sizeof(Tuple)* mNumElements;
	size_t subGroups = (numWorkGroups+mLocalSize-1)/mLocalSize;
	Event evKrnDataIn("Data Write");

	Timer* inst;
	//Timer timer;
	//timer.reset();
	double t;
	
	mTmpOutputBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, dataSizeInBytes, 0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create output buffer.");

	//create histogram buffer
	mHistogramBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(numWorkGroups+1), 0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create histogram buffer.");

	mTotalsBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(subGroups+1), 0, &err);
	CheckError(err, CL_SUCCESS, "Failed to create final output buffer.");



	if(isFirst) {

		//push data to accelerator
		
		//***Make sure to pass the address of the front of the vector!!!
		err = clEnqueueWriteBuffer(*(param.queue), param.memInput[0], CL_TRUE, 0, dataSizeInBytes,
				&(param.mInputVals[0].front()), 0, NULL, &evKrnDataIn.CLEvent());
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
	dataInSec = evKrnDataIn.StartEndRuntime()*nsToSec; 



	(param.resultDB)->AddResult("DataIn", sizeStr,"s", dataInSec);


	}

	return CL_SUCCESS;
}



int SelectionApp::RunKernel(bool isLast, BmkParams param) {
	cl_int err;
	
	size_t numGroups = mGlobalSize / mLocalSize;
	size_t subGroups = numGroups / mLocalSize;
	size_t mWgSz = 16;
	size_t dataSizeInBytes = sizeof(valType)* mNumElements;
	
	valType gpuResult = 0;
	Event evKrn1("Kernel 1");
	Event evKrn2("Kernel 2");
	Event evKrn2_1("Kernel 2_1");
	Event evKrn2_2("Kernel 2_2");
	Event evKrn3("Kernel 3");
	Event evKrnDataOut("Data Read");

	//Tuple* mHostOutput;
	//Currently Kernel 2, prefix sum must always use WG=256
	size_t preSumWgSz = 256;

	double time1Sec=0, time1_1Sec=0, time2Sec=0, time2_1Sec=0, time2_2Sec=0, time3Sec=0;

	Timer* inst;

	int start = inst->Start();
	
	//-------------Kernel 1-------------------------
	
	mWgSz =  FindKernelWorkGroupSize(mKernel1, param, mGlobalSize, mLocalSize);
	if(param.verbose)
		cout<< "K1: mGlobalSize "<<mGlobalSize<<" numGroups "<<numGroups<<" mLocalSize "<<mLocalSize<<" mWgSz "<<mWgSz<<endl;
	
	//execute the kernel.
	/*err = clEnqueueNDRangeKernel(*(param.queue), mKernel1, 1, NULL,
			&mGlobalSize,
			&mLocalSize, 0,
			NULL,
			&evKrn1.CLEvent());*/
	err = clEnqueueNDRangeKernel(*(param.queue), mKernel1, 1, NULL,
			&mGlobalSize,
			&mLocalSize, 0,
			NULL,
			&evKrn1.CLEvent());
	CL_CHECK_ERROR(err);

	err = clWaitForEvents (1, &evKrn1.CLEvent());
	CL_CHECK_ERROR(err);

	evKrn1.FillTimingInfo();
	if (param.verbose) evKrn1.Print(cerr);

	//-------------Kernel 2-------------------------

	//mWgSz =  FindKernelWorkGroupSize(mKernel2, param, mGlobalSize, numGroups);
	
	size_t lSize;
	//lSize = numGroups < preSumWgSz ? numGroups : preSumWgSz;
	lSize = numGroups < mLocalSize ? numGroups : mLocalSize;
	//lSize = 256;
	//size_t lSize = 1;

	if(param.verbose){
		cout<< "K2: mGlobalSize "<<mGlobalSize<<" numGroups "<<numGroups<<" mLocalSize "<<mLocalSize<<endl;//" mWgSz "<<mWgSz<<endli;
		cout<<" K2: lSize "<<lSize<<endl; }

	/*err = clEnqueueNDRangeKernel(*(param.queue), mKernel2, 1, NULL,
			&numGroups,
			&mWgSz, 0,
			NULL,
			&evKrn2.CLEvent());*/
	err = clEnqueueNDRangeKernel(*(param.queue), mKernel2, 1, NULL,
			&numGroups,
			&lSize, 0,
			NULL,
			&evKrn2.CLEvent());
	CL_CHECK_ERROR(err);

	err = clWaitForEvents (1, &evKrn2.CLEvent());
	CL_CHECK_ERROR(err);

	evKrn2.FillTimingInfo();
	if (param.verbose) evKrn2.Print(cerr);


	//-------------Kernel 2_1-------------------------
	size_t localSize; 
	if(subGroups > 1) {
		//localSize = subGroups < preSumWgSz ? subGroups : preSumWgSz;
		localSize = subGroups < mLocalSize ? subGroups : mLocalSize;
		//mWgSz =  FindKernelWorkGroupSize(mKernel2_1, param, subGroups, localSize);
		err = clEnqueueNDRangeKernel(*(param.queue), mKernel2_1, 1, NULL,
				&subGroups,
				&localSize, 0,
				NULL,
				&evKrn2_1.CLEvent());
		CL_CHECK_ERROR(err);

		err = clWaitForEvents (1, &evKrn2_1.CLEvent());
		CL_CHECK_ERROR(err);

		evKrn2_1.FillTimingInfo();
		if (param.verbose) evKrn2_1.Print(cerr);

		//-------------Kernel 2_2-------------------------
		err = clEnqueueNDRangeKernel(*(param.queue), mKernel2_2, 1, NULL,
				&numGroups,
				&mLocalSize, 0,
				NULL,
				&evKrn2_2.CLEvent());
		CL_CHECK_ERROR(err);

		err = clWaitForEvents (1, &evKrn2_2.CLEvent());
		CL_CHECK_ERROR(err);

		evKrn2_2.FillTimingInfo();
		if (param.verbose) evKrn2_2.Print(cerr);

	}

	//-------------Kernel 3-------------------------
	mWgSz =  FindKernelWorkGroupSize(mKernel3, param, mGlobalSize, mLocalSize);
	if(param.verbose)
		cout<< "K3: mGlobalSize "<<mGlobalSize<<" numGroups "<<numGroups<<" mLocalSize "<<mLocalSize<<" mWgSz "<<mWgSz<<endl;

	err = clEnqueueNDRangeKernel(*(param.queue), mKernel3, 1, NULL,
			&mGlobalSize,
			&mLocalSize, 0,
			NULL,
			&evKrn3.CLEvent());
	CL_CHECK_ERROR(err);

	err = clWaitForEvents (1, &evKrn3.CLEvent());
	CL_CHECK_ERROR(err);

	evKrn3.FillTimingInfo();
	if (param.verbose) evKrn3.Print(cerr);

	//----------Results---------------------------------
	char sizeStr[256];
	unsigned long long dataSizekB = (dataSizeInBytes)/1024;	
	//Filler until we handle multiple sizes 
	sprintf(sizeStr, "%7.llukB",dataSizekB);
	double nsToSec = 1.e-9;
	double dataInSec;

	time1Sec = evKrn1.StartEndRuntime()*nsToSec; 
	time2Sec = evKrn2.StartEndRuntime()*nsToSec; 

	(param.resultDB)->AddResult("SubKernel1", sizeStr,"s", time1Sec);
	(param.resultDB)->AddResult("SubKernel2", sizeStr,"s", time2Sec);


	if(subGroups > 1)
	{
		time2_1Sec = evKrn2_1.StartEndRuntime()*nsToSec; 
		time2_2Sec = evKrn2_2.StartEndRuntime()*nsToSec; 
		(param.resultDB)->AddResult("SubKernel2_1", sizeStr,"s", time2_1Sec);
		(param.resultDB)->AddResult("SubKernel2_2", sizeStr,"s", time2_2Sec);
	}
  
	time3Sec = evKrn3.StartEndRuntime()*nsToSec; 
	(param.resultDB)->AddResult("SubKernel3", sizeStr,"s", time3Sec);
  
	double krnl_time = time1Sec + time2Sec + time2_1Sec + time2_2Sec + time3Sec;
	(param.resultDB)->AddResult("SelectKernel", sizeStr,"s", krnl_time);

	
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

		//check if the CPU and GPU results match.
		int diffCount = 0;
		for(int i=0; i<mCpuOutput.size(); i++) {
			if (mCpuOutput[i].key != param.mOutputVals[i].key || mCpuOutput[i].valArray[0] != param.mOutputVals[i].valArray[0]) {
				if(param.verbose)
					printf("%d %u %u %u %u\n",i, mCpuOutput[i].key, mCpuOutput[i].valArray[0],param.mOutputVals[i].key, param.mOutputVals[i].valArray[0]);
				diffCount++;
			}
		}
		
		if(diffCount == 0)
		{
			if(param.verbose)
			Println("Verification outcome : PASSED!");
		}
		else
			Println("Verification outcome : FAILED!");

	
	err = clReleaseKernel(mKernel1);
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

void SelectionApp::Run(bool isLast, BmkParams param) {
	CheckError(SetKernel(param), CL_SUCCESS, "setKernel failed.");
	CheckError(RunKernel(isLast, param), CL_SUCCESS, "runKernel failed.");
}
