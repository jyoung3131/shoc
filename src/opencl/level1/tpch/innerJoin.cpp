#include "innerJoin.hpp"

//Reset the input and output vectors and existing cl_mem buffers
void InnerJoinApp::ResetBuffers()
{
}

//Currently no local device buffers for this benchmark
void InnerJoinApp::FreeDevBuffers()
{
    cl_int err;

    err = clReleaseMemObject(mOutputBuffer);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(mHistogramBuffer);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(mTotalsBuffer);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(mLowerBuffer);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(mUpperBuffer);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(mOutBoundsBuffer);
    CL_CHECK_ERROR(err);

}


void InnerJoinApp::SetSizes(size_t groupSize, size_t numLeftElements, size_t numRightElements, bool isFirstFlag)
{
        mLocalSize = groupSize;
        mGlobalSize = numLeftElements;
        mNumLeftElements = numLeftElements;
        mNumRightElements = numRightElements;
        isFirst = isFirstFlag;

        //cout<<"mLocalSize "<<mLocalSize<<" mGlobalSize "<<mGlobalSize<<" mNumLeftElements "<<mNumLeftElements<<" mNumRightElements "<<mNumRightElements<<" isFirst"<<isFirst<<endl;
}

size_t InnerJoinApp::RunCPUReference(double &t, vector<Tuple> left, vector<Tuple> right, size_t numLeftElements, size_t numRightElements, bool isFirst)
{
  mCpuOutput.clear();
  unsigned long long int outputSize = 0;
  //Timer timer;

  Timer* inst;

  int start = inst->Start();
  outputSize = 0;
  unsigned int l = 0, r = 0;

  while (l < numLeftElements && r < numRightElements) {

    valType lKey = left[l].key; 
    valType rKey = right[r].key;
    valType lElement = left[l].valArray[0];
    valType rElement;


        //cout<<"Left Elems: "<<numLeftElements<<" Right Elems: "<<numRightElements<<endl;
        //cout<<"LKey "<<lKey<<" RKey "<<rKey<<endl;

    
    if(lKey < rKey)
      ++l;
    else if(rKey < lKey)
      ++r;
    else {
      for(unsigned int i = r; i < numRightElements; ++i) {
        rKey = right[i].key;
        rElement = right[i].valArray[0];

        if(lKey < rKey) break;

        assert(lKey == rKey);
		Tuple tuple;
        tuple.key = lKey;
        if(isFirst){
          tuple.valArray[0] = lElement;
          tuple.valArray[1] = rElement;
        }
        else{
          tuple.valArray[0] = lElement;
          tuple.valArray[1] = left[l].valArray[1];
          tuple.valArray[2] = rElement;
        }
		mCpuOutput.push_back(tuple);
      }

      ++l;
    }
  }

  t = t + (inst->Stop(start, "CPU"));

  //Used for debugging purposes

  /*	
  for(int i = 0; i < left.size(); i++)
	  cout<<"["<<i<<"]: left="<<left[i].key<<", "<<left[i].valArray[0]<<", "<<left[i].valArray[1]<<endl;
  
  for(int i = 0; i < right.size(); i++)
	  cout<<"["<<i<<"]: right="<<right[i].key<<", "<<right[i].valArray[0]<<", "<<right[i].valArray[1]<<endl;

  for(int i = 0; i < mCpuOutput.size(); i++)
  {
	cout<<"["<<i<<"]: cpu="<<mCpuOutput[i].key<<", "<<mCpuOutput[i].valArray[0]<<", "<<mCpuOutput[i].valArray[1]<<endl;
  }*/

  return mCpuOutput.size();
}


int InnerJoinApp::SetKernel(BmkParams param)
{
  cl_int err;

  //Finds the bounds of the inputs (if left and right elements differe) and
  //output partitions that fit within a workgroup
  mKernel1 = clCreateKernel(program, "FindBounds", &err);
  CL_CHECK_ERROR(err);

  //Perform a prefix sum over the bounds array and determine if
  //subgroups (and kernel 2_1 and 2_2) are needed
  mKernel2 = clCreateKernel(program, "PrefixSum", &err);
  CL_CHECK_ERROR(err);

  //Both of these kernels are only used if subGroups are needed
  mKernel2_1 = clCreateKernel(program, "PrefixSum", &err);
  CL_CHECK_ERROR(err);
    
mKernel2_2 = clCreateKernel(program, "Sum", &err);
  CL_CHECK_ERROR(err);

    mKernel3 = clCreateKernel(program, "Join", &err);
  CL_CHECK_ERROR(err);

    mKernel4 = clCreateKernel(program, "PrefixSum", &err);
  CL_CHECK_ERROR(err);

    mKernel4_1 = clCreateKernel(program, "PrefixSum", &err);
  CL_CHECK_ERROR(err);

    mKernel4_2 = clCreateKernel(program, "Sum", &err);
  CL_CHECK_ERROR(err);

    mKernel5 = clCreateKernel(program, "GatherJoin", &err);
  CL_CHECK_ERROR(err);

  unsigned int groups = (mGlobalSize+mLocalSize-1)/mLocalSize;
  unsigned int subGroups = (groups+mLocalSize-1)/mLocalSize;
  unsigned int partitionSize = (mNumLeftElements + groups - 1) / groups;
  unsigned int isTotal = 0;
  if(subGroups > 1) isTotal = 1; 
  unsigned int noTotal = 0;
  unsigned int first = 0;
  if(isFirst) first = 1;
  //set kernel arguments

	err = clSetKernelArg(mKernel1, 0, sizeof(cl_mem), &param.memInput[0]);
  CL_CHECK_ERROR(err);
	err += clSetKernelArg(mKernel1, 1, sizeof(cl_mem),&param.memInput[1]);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel1, 2, sizeof(cl_mem), &mLowerBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel1, 3, sizeof(cl_mem), &mUpperBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel1, 4, sizeof(cl_mem), &mOutBoundsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel1, 5, sizeof(unsigned int), &mNumLeftElements);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel1, 6, sizeof(unsigned int), &mNumRightElements);
  CL_CHECK_ERROR(err);
    //err += clSetKernelArg(mKernel1, 7, sizeof(cl_mem), mLeftRankBuffer);
  //CL_CHECK_ERROR(err);
    //err += clSetKernelArg(mKernel1, 8, sizeof(cl_mem), mRightRankBuffer);
  //CL_CHECK_ERROR(err);


    err += clSetKernelArg(mKernel2, 0, sizeof(cl_mem), &mOutBoundsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel2, 1, sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel2, 2,sizeof(unsigned int),&isTotal);
  CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel2_1, 0,sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel2_1, 1,sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel2_1, 2,sizeof(unsigned int),&noTotal);
  CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel2_2, 0,sizeof(cl_mem), &mOutBoundsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel2_2, 1,sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);

    err = clSetKernelArg(mKernel3, 0, sizeof(cl_mem), &param.memInput[0]);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 1, sizeof(cl_mem), &param.memInput[1]);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 2, sizeof(cl_mem), &mLowerBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 3, sizeof(cl_mem), &mUpperBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 4, sizeof(cl_mem), &mOutBoundsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 5, sizeof(unsigned int), &mNumLeftElements);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 6, sizeof(unsigned int), &mNumRightElements);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 7, sizeof(cl_mem), &mOutputBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 8, sizeof(cl_mem), &mHistogramBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel3, 9, sizeof(unsigned int), &first);
  CL_CHECK_ERROR(err);
    //err += clSetKernelArg(mKernel3, 9, sizeof(cl_mem), &mLeftRankBuffer);
    //err += clSetKernelArg(mKernel3, 10, sizeof(cl_mem), &mRightRankBuffer);

    err += clSetKernelArg(mKernel4, 0,sizeof(cl_mem), &mHistogramBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel4, 1,sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel4, 2,sizeof(unsigned int),&isTotal);
  CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel4_1, 0,sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel4_1, 1,sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel4_1, 2,sizeof(unsigned int),&noTotal);
  CL_CHECK_ERROR(err);

    err += clSetKernelArg(mKernel4_2, 0,sizeof(cl_mem), &mHistogramBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel4_2, 1,sizeof(cl_mem), &mTotalsBuffer);
  CL_CHECK_ERROR(err);

	err += clSetKernelArg(mKernel5, 0, sizeof(cl_mem),&param.memOutput);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel5, 1,sizeof(cl_mem), &mOutputBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel5, 2,sizeof(cl_mem), &mOutBoundsBuffer);
  CL_CHECK_ERROR(err);
    err += clSetKernelArg(mKernel5, 3,sizeof(cl_mem), &mHistogramBuffer);
  CL_CHECK_ERROR(err);

  return CL_SUCCESS;
}

int InnerJoinApp::SetBuffers(BmkParams param)
{
  cl_int err;
  size_t numWorkGroups = (mGlobalSize+mLocalSize-1) / mLocalSize;
  size_t leftDataSizeInBytes = sizeof(Tuple)* mNumLeftElements;
  size_t rightDataSizeInBytes = sizeof(Tuple)* mNumRightElements;
  size_t inputDataSizeInBytes = leftDataSizeInBytes + rightDataSizeInBytes;
  size_t subGroups = (numWorkGroups+mLocalSize-1)/mLocalSize;

  //cout<<"Set Buffers: numWG "<<numWorkGroups<<" LDataSzB "<<leftDataSizeInBytes<<" RDataSzB "<<rightDataSizeInBytes<<" subGroups "<<subGroups<<endl;

  Event evKrnDataIn1("Data1 Write");
  Event evKrnDataIn2("Data2 Write");

  unsigned int zeroPattern = 0; 
    
  //Use a buffer that is 1.5 times bigger than the output
  mOutputBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, (size_t)(param.cpuOutputSzBytes+sizeof(Tuple)), NULL, &err);
  //printf("Size of tuple is %d\n", sizeof(Tuple));
  //134217728 is the max for Trinity GPU
  //mOutputBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(Tuple)*100000, NULL, &err);
  CheckError(err, CL_SUCCESS, "Failed to create output buffer.");
  err = clEnqueueFillBuffer(*(param.queue), mOutputBuffer, &zeroPattern, sizeof(unsigned int), 0, (size_t)(param.cpuOutputSzBytes+sizeof(Tuple)), 0, NULL, NULL);
  CL_CHECK_ERROR(err);

  //create histogram buffer
  mOutBoundsBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(numWorkGroups+1), 0, &err);
  CheckError(err, CL_SUCCESS, "Failed to create outBounds buffer.");
  err = clEnqueueFillBuffer(*(param.queue), mOutBoundsBuffer, &zeroPattern, sizeof(unsigned int), 0, sizeof(unsigned int)*(numWorkGroups+1), 0, NULL, NULL);
  CL_CHECK_ERROR(err);

  mHistogramBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(numWorkGroups+1), 0, &err);
  err = clEnqueueFillBuffer(*(param.queue), mHistogramBuffer, &zeroPattern, sizeof(unsigned int), 0, sizeof(unsigned int)*(numWorkGroups+1), 0, NULL, NULL);
  CL_CHECK_ERROR(err);

  mUpperBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(numWorkGroups), 0, &err);
  CheckError(err, CL_SUCCESS, "Failed to create upper buffer.");
  err = clEnqueueFillBuffer(*(param.queue), mUpperBuffer, &zeroPattern, sizeof(unsigned int), 0, sizeof(unsigned int)*(numWorkGroups), 0, NULL, NULL);
  CL_CHECK_ERROR(err);

  mLowerBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(numWorkGroups), 0, &err);
  CheckError(err, CL_SUCCESS, "Failed to create lower buffer.");
  err = clEnqueueFillBuffer(*(param.queue), mLowerBuffer, &zeroPattern, sizeof(unsigned int), 0, sizeof(unsigned int)*(numWorkGroups), 0, NULL, NULL);
  CL_CHECK_ERROR(err);

  mTotalsBuffer = clCreateBuffer(*(param.ctx), CL_MEM_READ_WRITE, sizeof(unsigned int)*(subGroups+1), 0, &err);
  CheckError(err, CL_SUCCESS, "Failed to create totals buffer.");
  err = clEnqueueFillBuffer(*(param.queue), mTotalsBuffer, &zeroPattern, sizeof(unsigned int), 0, sizeof(unsigned int)*(subGroups+1), 0, NULL, NULL);
  CL_CHECK_ERROR(err);



	err = clEnqueueWriteBuffer(*(param.queue), param.memInput[0], CL_TRUE, 0, leftDataSizeInBytes,
				&(param.mInputVals[0].front()), 0, NULL, &evKrnDataIn1.CLEvent());
	CL_CHECK_ERROR(err);
	
    err = clWaitForEvents (1, &evKrnDataIn1.CLEvent());
	CL_CHECK_ERROR(err);
	
    evKrnDataIn1.FillTimingInfo();
	if (param.verbose) evKrnDataIn1.Print(cerr);
    
	err = clEnqueueWriteBuffer(*(param.queue), param.memInput[1], CL_TRUE, 0, rightDataSizeInBytes,
				&(param.mInputVals[1].front()), 0, NULL, &evKrnDataIn2.CLEvent());
	CL_CHECK_ERROR(err);
	
    err = clWaitForEvents (1, &evKrnDataIn2.CLEvent());
	CL_CHECK_ERROR(err);
    
    evKrnDataIn2.FillTimingInfo();
	if (param.verbose) evKrnDataIn2.Print(cerr);

	double dataInNs = 0;
	dataInNs = evKrnDataIn1.StartEndRuntime(); 
	dataInNs += evKrnDataIn2.StartEndRuntime(); 
	AddTimeResultToDB(param, dataInNs, inputDataSizeInBytes, "DataIn");

  return CL_SUCCESS;
}

int InnerJoinApp::RunKernel(BmkParams param)
{
  cl_int err;
  size_t numGroups = (mGlobalSize+mLocalSize-1) / mLocalSize;
  size_t subGroups = (numGroups+mLocalSize-1)/mLocalSize;
  valType gpuResult = 0;
  Event evKrn1("Kernel 1");
  Event evKrn2("Kernel 2");
  Event evKrn2_1("Kernel 2_1");
  Event evKrn2_2("Kernel 2_2");
  Event evKrn3("Kernel3");
  Event evKrn4("Kernel 4");
  Event evKrn4_1("Kernel 4_1");
  Event evKrn4_2("Kernel 4_2");
  Event evKrn5("Kernel 5");
  Event evDataRd("DataOut");
  double cpuTime;
  Tuple *mapPtr;
  valType *mapPtr1;
  valType *mapPtr2;
  unsigned int* mapPtr3;
  unsigned long long dataSizeInBytes = mNumLeftElements*2*sizeof(Tuple);	
  unsigned long long outputSizeInBytes = param.cpuOutputSzBytes+sizeof(Tuple);	
  unsigned long long outputNumElems = param.cpuOutputSzBytes/sizeof(Tuple);	

  double time1Sec=0, time2Sec=0, time2_1Sec=0, time2_2Sec=0, time3Sec=0, time4Sec=0, time4_1Sec=0, time4_2Sec=0, time5Sec=0;


  //execute the kernel.
  //TODO: Should mLocalSize be set using GetKernelWorkGroupInfo
  // Determine the maximum work group size for this kernel
  //  maxGroupSize = getMaxWorkGroupSize(dev);

    //-------------Kernel 1-------------------------
    //cout<<"mLocalSize = "<<mLocalSize<<" global size"<<mGlobalSize<<endl;

    //Make sure that the local work item size is divisible by the global size
    //if(mGlobalSize < mLocalSize)
	//    mLocalSize = mGlobalSize;

  /*DebugDevMem(param, param.memInput[0], mNumLeftElements, true,"Ltuple_k1");
  DebugDevMem(param, param.memInput[1], mNumRightElements, true,"Rtuple_k1");
  DebugDevMem(param, param.memOutput, outputNumElems, true,"OutTuple_k1");
  DebugDevMem(param, mOutputBuffer, outputNumElems, true,"TmpOutTuple_k1");
  DebugDevMem(param, mOutBoundsBuffer, numGroups+1, false,"OutBounds_k1");
  DebugDevMem(param, mLowerBuffer, numGroups, false,"LowerBuff_k1");
  DebugDevMem(param, mUpperBuffer, numGroups, false,"UpperBuff_k1");
  DebugDevMem(param, mHistogramBuffer, numGroups+1, false,"HistogramBuff_k1");
  DebugDevMem(param, mTotalsBuffer, subGroups+1, false,"TotalsBuff_k1");*/
  err = clEnqueueNDRangeKernel(*(param.queue), mKernel1, 1, NULL, &mGlobalSize, &mLocalSize, 0, NULL, &evKrn1.CLEvent());
  CL_CHECK_ERROR(err);

  err = clWaitForEvents (1, &evKrn1.CLEvent());
  CL_CHECK_ERROR(err);

/*  DebugDevMem(param, mOutBoundsBuffer, numGroups+1, false,"OutBounds_post_k1");
  DebugDevMem(param, mLowerBuffer, numGroups, false,"LowerBuff_post_k1");
  DebugDevMem(param, mUpperBuffer, numGroups, false,"UpperBuff_post_k1");*/
  evKrn1.FillTimingInfo();
  if (param.verbose) evKrn1.Print(cerr);

  //-----------------Kernel 2----------------------------------

  size_t lSize = numGroups < mLocalSize ? numGroups : mLocalSize;

   
  //cout<<"lSize = "<<lSize<<" numGroups"<<numGroups<<endl;
  err = clEnqueueNDRangeKernel(*(param.queue), mKernel2, 1, NULL,
      &numGroups,
      &lSize, 0,
      NULL,
      &evKrn2.CLEvent());
  CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");


  err = clWaitForEvents (1, &evKrn2.CLEvent());
  CL_CHECK_ERROR(err);
  evKrn2.FillTimingInfo();
  if (param.verbose) evKrn2.Print(cerr);
/*  DebugDevMem(param, mOutBoundsBuffer, numGroups+1, false,"OutBounds_k2");
  DebugDevMem(param, mTotalsBuffer, subGroups+1, false,"TotalsBuff_k2");*/

  //-----------------Kernel 2_1----------------------------------
  if(subGroups > 1){
    size_t localSize = subGroups < mLocalSize ? subGroups : mLocalSize;
    err = clEnqueueNDRangeKernel(*(param.queue), mKernel2_1, 1, NULL,
        &subGroups, 
        &localSize, 0,
        //&numGroups,
        NULL,
        &evKrn2_1.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");


    err = clWaitForEvents (1, &evKrn2_1.CLEvent());
    CL_CHECK_ERROR(err);
    evKrn2_1.FillTimingInfo();
    if (param.verbose) evKrn2_1.Print(cerr);


  //-----------------Kernel 2_2----------------------------------

    err = clEnqueueNDRangeKernel(*(param.queue), mKernel2_2, 1, NULL,
        &numGroups, 
        &mLocalSize, 0,
        NULL,
        &evKrn2_2.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");


    err = clWaitForEvents (1, &evKrn2_2.CLEvent());
    CL_CHECK_ERROR(err);
    evKrn2_2.FillTimingInfo();
    if (param.verbose) evKrn2_2.Print(cerr);

  }
  clFinish(*(param.queue));

  //-----------------Kernel 3----------------------------------
  err = clEnqueueNDRangeKernel(*(param.queue), mKernel3, 1, NULL,
      &mGlobalSize,
      &mLocalSize, 0,
      NULL,
      &evKrn3.CLEvent());
  CL_CHECK_ERROR(err);

  err = clWaitForEvents (1, &evKrn3.CLEvent());
  CL_CHECK_ERROR(err);
  
  clFinish(*(param.queue));
/*  DebugDevMem(param, mOutputBuffer, outputNumElems, true,"TmpOutTuple_k3");
  DebugDevMem(param, mOutBoundsBuffer, numGroups+1, false,"OutBounds_k3");
  DebugDevMem(param, mLowerBuffer, numGroups, false,"LowerBuff_k3");
  DebugDevMem(param, mUpperBuffer, numGroups, false,"UpperBuff_k3");
  DebugDevMem(param, mHistogramBuffer, numGroups+1, false,"HistogramBuff_k3");*/
  
  evKrn3.FillTimingInfo();
  if (param.verbose) evKrn3.Print(cerr);
  


  //-----------------Kernel 4----------------------------------
  err = clEnqueueNDRangeKernel(*(param.queue), mKernel4, 1, NULL,
      &numGroups,
      &lSize, 0,
      NULL,
      &evKrn4.CLEvent());
  CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");

  err = clWaitForEvents (1, &evKrn4.CLEvent());
  CL_CHECK_ERROR(err);
  evKrn4.FillTimingInfo();
  if (param.verbose) evKrn4.Print(cerr);
  
/*  DebugDevMem(param, mHistogramBuffer, numGroups+1, false,"HistogramBuff_k4");
  DebugDevMem(param, mTotalsBuffer, subGroups+1, false,"TotalsBuff_k4");*/

  //-----------------Kernel 4_1----------------------------------
  if(subGroups > 1){
    size_t localSize = subGroups < mLocalSize ? subGroups : mLocalSize;
    err = clEnqueueNDRangeKernel(*(param.queue), mKernel4_1, 1, NULL,
        &subGroups, 
        &localSize, 0,
        NULL,
        &evKrn4_1.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");

    err = clWaitForEvents (1, &evKrn4_1.CLEvent());
    CL_CHECK_ERROR(err);
    evKrn4_1.FillTimingInfo();
    if (param.verbose) evKrn4_1.Print(cerr);

  //-----------------Kernel 4_2----------------------------------
    err = clEnqueueNDRangeKernel(*(param.queue), mKernel4_2, 1, NULL,
        &numGroups, 
        &mLocalSize,
        0,
        NULL,
        &evKrn4_2.CLEvent());
    CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");

    err = clWaitForEvents (1, &evKrn4_2.CLEvent());
    CL_CHECK_ERROR(err);
    evKrn4_2.FillTimingInfo();
    if (param.verbose) evKrn4_2.Print(cerr);


  }
  
  //-----------------Kernel 5----------------------------------

  //cout<<"Kernel 5"<<endl;

  err = clEnqueueNDRangeKernel(*(param.queue), mKernel5, 1, NULL,
      &mGlobalSize,
      &mLocalSize, 0,
      NULL,
      &evKrn5.CLEvent());
  //CheckError(err, CL_SUCCESS, "Failed to enqueue kernel.");
  CL_CHECK_ERROR(err);


  err = clWaitForEvents (1, &evKrn5.CLEvent());
  CL_CHECK_ERROR(err);
  evKrn5.FillTimingInfo();
  if (param.verbose) evKrn5.Print(cerr);


  double nsToSec = 1.e-9;
  time1Sec = evKrn1.StartEndRuntime()*nsToSec; 
  time2Sec = evKrn2.StartEndRuntime()*nsToSec; 
  
/*  DebugDevMem(param, mOutputBuffer, outputNumElems, true,"TmpOutTuple_k5");
  DebugDevMem(param, mOutBoundsBuffer, numGroups+1, false,"OutBounds_k5");
  DebugDevMem(param, mHistogramBuffer, numGroups+1, false,"HistogramBuff_k5");*/
	
  //----------Kernel Timing Results------------------------------
    //AddTimeResultToDB(param, evKrn1.StartEndRuntime(), dataSizeInBytes, "Kernel 1");	

  clFinish(*(param.queue));
  
  char sizeStr[256];
  unsigned long long dataSizekB = (mNumLeftElements*16*2)/2014;	
  sprintf(sizeStr, "%7llukB",dataSizekB);

  //TODO: Map this to multiple input sizes
  //sprintf(sizeStr, "% 7dkB", sizes[sizeIndex]);


  (param.resultDB)->AddResult("SubKernel1", sizeStr,"s", time1Sec);
  (param.resultDB)->AddResult("SubKernel2", sizeStr,"s", time2Sec);

  if(subGroups > 1){
    time2_1Sec = evKrn2_1.StartEndRuntime()*nsToSec; 
    time2_2Sec = evKrn2_2.StartEndRuntime()*nsToSec; 
  (param.resultDB)->AddResult("SubKernel2_1", sizeStr,"s", time2_1Sec);
  (param.resultDB)->AddResult("SubKernel2_2", sizeStr,"s", time2_2Sec);
  }

  time3Sec = evKrn3.StartEndRuntime()*nsToSec; 
  time4Sec = evKrn4.StartEndRuntime()*nsToSec; 
  (param.resultDB)->AddResult("SubKernel3", sizeStr,"s", time3Sec);
  (param.resultDB)->AddResult("SubKernel4", sizeStr,"s", time4Sec);

  if(subGroups > 1){
  time4_1Sec = evKrn4_1.StartEndRuntime()*nsToSec; 
  time4_2Sec = evKrn4_2.StartEndRuntime()*nsToSec; 
  (param.resultDB)->AddResult("SubKernel4_1", sizeStr,"s", time4_1Sec);
  (param.resultDB)->AddResult("SubKernel4_2", sizeStr,"s", time4_2Sec);
  }

  time5Sec = evKrn5.StartEndRuntime()*nsToSec;
  (param.resultDB)->AddResult("SubKernel5", sizeStr,"s", time5Sec);

  double krnl_time = time1Sec + time2Sec + time2_1Sec + time2_2Sec + time3Sec + time4Sec + time4_1Sec + time4_2Sec+ time5Sec;
  (param.resultDB)->AddResult("JoinKernel", sizeStr,"s", krnl_time);


  //----------Transfer Data Out---------------------------
  /*mapPtr = (Tuple*)clEnqueueMapBuffer(*(param.queue), mFinalOutBuffer, CL_TRUE, CL_MAP_READ, 0,
      sizeof(Tuple)* (2*MAX(mNumLeftElements,mNumRightElements)), 0, NULL, NULL, &err);
  CheckError(err, CL_SUCCESS, "Failed to map output buffer.");*/

	err = clEnqueueReadBuffer(*(param.queue), param.memOutput, CL_TRUE, 0,
			 outputSizeInBytes,
				&param.mOutputVals.front(), 0, NULL, &evDataRd.CLEvent());
    CL_CHECK_ERROR(err);

	err = clWaitForEvents (1, &evDataRd.CLEvent());
	CL_CHECK_ERROR(err);
    
    evDataRd.FillTimingInfo();
    if (param.verbose) evDataRd.Print(cerr);

    AddTimeResultToDB(param, evDataRd.StartEndRuntime(), dataSizeInBytes, "Data Out");

  //Clear the queue
  clFinish(*(param.queue));
  //----------Validation----------------------------------
  //check if the CPU and GPU results match.
  int diffCount = VerifyResults(param, mCpuOutput, param.mOutputVals);

  diffCount = 0;
  //check if the CPU and GPU results match.
  /*for(int i=0;i<mCpuOutput.size();i++){
	//if ((mCpuOutput[i].key != param.mOutputVals[i].key) && (mCpuOutput[i].valArray[0] != param.mOutputVals[i].valArray[0]))
           // {
                cout<<"["<<i<<"]: hcpu="<<mCpuOutput[i].key<<", "<<mCpuOutput[i].valArray[0]<<", "<<mCpuOutput[i].valArray[1]<<endl;
                cout<<"["<<i<<"]: haccel="<<param.mOutputVals[i].key<<", "<<param.mOutputVals[i].valArray[0]<<", "<<param.mOutputVals[i].valArray[1]<<endl;
             //   diffCount++;
            //}

	}*/

  if(diffCount == 0)
  {
    if(param.verbose)
    	cout<<"Verification outcome : PASSED!"<<endl; 
  }
  else
    Println("Verification outcome : FAILED!"); 


    //Release the kernels
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
	
    err = clReleaseKernel(mKernel4);
	CL_CHECK_ERROR(err);

    err = clReleaseKernel(mKernel4_1);
	CL_CHECK_ERROR(err);
	
    err = clReleaseKernel(mKernel4_2);
	CL_CHECK_ERROR(err);
	
    err = clReleaseKernel(mKernel5);
	CL_CHECK_ERROR(err);

  return CL_SUCCESS;
}

void InnerJoinApp::Run(BmkParams param)
{
  CheckError(SetKernel(param), CL_SUCCESS, "setKernel failed.");
  CheckError(RunKernel(param), CL_SUCCESS, "runKernel failed.");
}

