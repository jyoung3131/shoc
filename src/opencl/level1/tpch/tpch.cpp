#include "tpch.h"
#include <iostream>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "Timer.h"

#include "innerJoin.hpp"
#include "select.hpp"
#include "product.hpp"
#include "project.hpp"
#include "unique.hpp"

/*TODO: Remove the requirement for a global program variable*/

cl_program program;
// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeff Young
// Creation: August 1, 2014
//
// Modifications:
//
// ****************************************************************************

void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("max-mb", OPT_INT, "1", "Data size (in megabytes)");
    op.addOption("num-elems", OPT_INT, "0", "Data size (num elements)");
    op.addOption("test-name", OPT_STRING, "Select", "Pass specific test to run"); 
    op.addOption("join-size", OPT_INT, "1", "Specify size range of join output,1-4 with 4 as the largest"); 
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes a series of TPC-H primitive and microbenchmarks for OpenCL devices.
//    
//
// Arguments:
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this DB
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Ifrah Saeed, Jeff Young
// Creation: August 1, 2014
//
// Modifications: Meghana Gupta, Jeff Young - August 29, 2014 Updated for SHOC compatibility and renamed tests
//
// ****************************************************************************
void RunBenchmark(cl_device_id dev,
        cl_context ctx,
        cl_command_queue queue,
        ResultDatabase &resultDB,
        OptionParser &op)
{
    //Initialize random num generator
    srand((unsigned)time(NULL));

    //Standard SHOC options:
    //---------------------
    const bool waitForEvents = true;
    bool verbose = op.getOptionBool("verbose");
    bool quiet = op.getOptionBool("quiet");
    int npasses = op.getOptionInt("passes");
    int joinSz = op.getOptionInt("join-size");


    //Define a struct that contains the device  and platform 
    //information as well as parameters for this benchmark
    BmkParams shocParams(&dev, &ctx, &queue, &resultDB, verbose, quiet);

    if(npasses != 0)
    {
        shocParams.nPasses = npasses;
    }
    else
        npasses = 1;

    //
    shocParams.joinSize = joinSz;

    //Tuple size is 8-16 B, so max MB option should be divided by this
    //Also, some primitives use a single tuple while others use multiple (e.g., Join)
    //With 16 B tuples, this array goes up to 2 GB per input - it's unlikely all sizes will be used
    //for most inputs

    /*NOTE: Anything less than 256 elems will not work with Join!*/
    unsigned long long inputElemSizes[27] = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
        32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,
        67108864,134217728};


    //User-specified options:
    //-----------------------
    //Total number of passes per size * total number of sizes
    int totalRuns;

    bool singleTest = false; 
    size_t userNumElems;
    userNumElems = op.getOptionInt("num-elems");
    shocParams.SetNumElements(userNumElems);

    //Only one size will be tested if num-elems is specified
    if(shocParams.numElems != 0)
        singleTest = true;

    string testName = op.getOptionString("test-name");
    if(testName != "")	
        shocParams.testName = testName;	
    else
        testName = "tpch";

    string outName = testName + ".csv";

    //TODO - rename this option at some point
    unsigned long maxInputSizeMB = op.getOptionInt("max-mb");

    //Calculated values
    //-----------------
    //Specify the range of input element sizes to use
    int minSzIdx, maxSzIdx;

    //Set the OpenCL file that contains all OpenCL primitives
    CheckError(SetProgram("tpch.cl", ctx, dev), CL_SUCCESS, "setProgram failed.");

    //Get device-specific values like max allocation size
    GetDeviceSettings(&shocParams);

    //TODO - determine maximum size to use; it depends on max input allocation 
    //for this device; if num-elems is specified, then minIdx=maxIdx 
    SetTestRange(&shocParams, &minSzIdx, &maxSzIdx);
    totalRuns = ((maxSzIdx-minSzIdx) + 1)*npasses;

    if(verbose)
    {
        cout<<"Testing from range "<<minSzIdx<<" to "<<maxSzIdx<<endl
            <<" numElems "<<userNumElems<<" totalRuns "<<totalRuns<<endl;  
    }

    //If verbose or quiet is not specified, use the ProgressBar to indicate 
    //the benchmark's progress
    ProgressBar pb(totalRuns);
    if (!verbose && !quiet)
        pb.Show(stdout);


    //Formatted string to print out size to file	
    char sizeStr[256] = {};
    char passStr[256] = {};
    unsigned long long dataSizekB;
    
    //Add the device name to the beginning of any new results run within the same file
    shocParams.resultDB->AddResult("**"+shocParams.devName, sizeStr, passStr, FLT_MAX);
    //Number of iterations used for this test
    sprintf(passStr, "%5d iter ",npasses);
        

    for(int i = minSzIdx; i <= maxSzIdx; i++)
    {
        if(!singleTest)
            shocParams.SetNumElements(inputElemSizes[i]);

        dataSizekB = (inputElemSizes[i]*sizeof(Tuple))/1024;		
        sprintf(sizeStr, "%7llukB",dataSizekB);
        shocParams.resultDB->AddResult("*"+shocParams.testName, sizeStr,passStr, FLT_MAX);

        //npasses are run within the RunTest function
	RunTest(&shocParams, npasses, &pb);

        //Print out the results to a file for each size
        //ResultsDB averages data over multiple passes
        shocParams.resultDB->DumpCsv(outName);

        //Allow for separate tests to be placed in the same file
        shocParams.resultDB->ClearAllResults();
    }

    cout<<endl<<endl<<"***Please check output file "<<outName<<" for results***"<<endl<<endl;


}

void RunTest(BmkParams* param, int npasses, ProgressBar* pb)
{

    if(param->verbose)
    {
        cout<<"***Testing "<<param->testName<<"***"<<endl;
        cout<<"Testing for size "<<param->numElems<<endl;
    }
    //TODO - remove this as it is only needed to handle prefix sum
    size_t localSize = 256;  
    valType threshold; //get it as an input
    double cpuTime = 0;


    //For single input tests
    //Clear and reinitialize the input vector
    //InitializeHostVector(&(param->mInputVals[0]), param->numElems);
    //Clear the output vector
    //ResetHostVector(&param->mOutputVals);

    if(param->testName == "Select")
    {
        SelectionApp *selection;
        selection = new SelectionApp();

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Create input and output buffers on the host

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Output
        AllocateDevBuffer(&param->memOutput, (double)(param->devBufferSz), 1, *(param->ctx));


        threshold = (size_t)(param->numElems/2);
        selection->SetSizes(localSize, (size_t)param->numElems, threshold); //give threshold as input


        //Use the same input and output buffers 	
        for(int j = 0; j < npasses; j++)
        {
            //For single input tests
            //Clear and reinitialize the input vector
            InitializeHostVector(&(param->mInputVals[0]), param->numElems, false, param->joinSize);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize(param->numElems);

            selection->RunCPUReference(cpuTime, param->mInputVals[0], param->numElems);

            if(param->verbose)
                Println("CPU : [%f secs]", cpuTime);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(selection->SetBuffers(1, *param), CL_SUCCESS, "setBuffers failed.");
            selection->Run(1, *param);			

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            selection->FreeDevBuffers();	

            // update progress bar
            pb->addItersDone();
            if (!param->verbose && !param->quiet)
                pb->Show(stdout);

        }//End npasses


        //Free both buffers
        DeallocateDevBuffer(param->memInput[0]);
        DeallocateDevBuffer(param->memOutput);	

        free(selection);

    }
    else if(param->testName == "Product")
    {
        /*ProductApp *product;
        product = new ProductApp();

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Create input and output buffers on the host

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Input 2 
        AllocateDevBuffer(&param->memInput[1], (double)(param->devBufferSz), 0, *(param->ctx));
        //Output - for product can be (input)^2
        AllocateDevBuffer(&param->memOutput, (double)((param->devBufferSz)*(param->devBufferSz)), 1, *(param->ctx));

        //Right now we assume right and left hand sides have equal numbers of elements
        product->SetSizes(localSize, (size_t)param->numElems, (size_t)param->numElems); //give threshold as input


        //Use the same input and output buffers 	
        for(int j = 0; j < npasses; j++)
        {
            //For single input tests
            //Clear and reinitialize the input vectors
            InitializeHostVector(&(param->mInputVals[0]), param->numElems, false);
            InitializeHostVector(&(param->mInputVals[1]), param->numElems, false);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize((param->numElems*param->numElems));

            product->RunCPUReference(cpuTime, param->mInputVals[0], param->mInputVals[1], param->numElems, param->numElems);

            if(param->verbose)
                Println("CPU : [%f secs]", cpuTime);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(product->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
            product->Run(*param);			

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            //product->FreeDevBuffers();	


            // update progress bar
            pb->addItersDone();
            if (!param->verbose && !param->quiet)
                pb->Show(stdout);

        }//End npasses


        //Free device buffers
        DeallocateDevBuffer(param->memInput[0]);
        DeallocateDevBuffer(param->memInput[1]);
        DeallocateDevBuffer(param->memOutput);	

        free(product);*/

    }//end Product
    else if(param->testName == "Join")
    {
        for(int j = 0; j < npasses; j++)
        {
        InnerJoinApp *join;
        join = new InnerJoinApp();

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Create input and output buffers on the host

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Input 2 
        AllocateDevBuffer(&param->memInput[1], (double)(param->devBufferSz), 0, *(param->ctx));
        //Output - for join can be (input)^2
        //TODO - check the sizing of this buffer as it seems like it might not be right.

        //Right now we assume right and left hand sides have equal numbers of elements
        join->SetSizes(localSize, (size_t)param->numElems, (size_t)param->numElems, true); //give threshold as input

        size_t joinOutputSz;

        //Use the same input and output buffers 	
        /*for(int j = 0; j < npasses; j++)
        {*/
            //For single input tests
            //Clear and reinitialize the input vectors
            //*Note here that we are sorting the input data set for the join operation 
            InitializeHostVector(&(param->mInputVals[0]), param->numElems, true, param->joinSize);
            InitializeHostVector(&(param->mInputVals[1]), param->numElems, true, param->joinSize);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);

            //Here we are cheating slightly by getting the output size based on the
            //CPU test and using it to allocate output buffers
            joinOutputSz = join->RunCPUReference(cpuTime, param->mInputVals[0], param->mInputVals[1], param->numElems, param->numElems, true);
	    
	    param->cpuOutputSzBytes = joinOutputSz*sizeof(Tuple);
            if(param->verbose)
                Println("CPU : [%f secs]; Output Size: [%d]; OutputBytes: [%d]", cpuTime, joinOutputSz, param->cpuOutputSzBytes);
            
            
            
            //Resize the output vector to hold new outputs
            param->mOutputVals.resize(joinOutputSz+sizeof(Tuple));
            AllocateDevBuffer(&param->memOutput, (double)(param->cpuOutputSzBytes*2), 1, *(param->ctx));
	    ZeroDevBuffer(*param, param->memOutput, param->cpuOutputSzBytes);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(join->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
            join->Run(*param);			

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            join->FreeDevBuffers();	
            DeallocateDevBuffer(param->memOutput);	

            // update progress bar
            pb->addItersDone();
            if (!param->verbose && !param->quiet)
                pb->Show(stdout);

 //       }//End npasses


        //Free device buffers
        DeallocateDevBuffer(param->memInput[0]);
        DeallocateDevBuffer(param->memInput[1]);

        free(join);

	}//temp end npasses for debugging

    }//end Join
    else if(param->testName == "Unique")
    {

        UniqueApp *unique;
        unique = new UniqueApp();

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Output
        AllocateDevBuffer(&param->memOutput, (double)(param->devBufferSz), 1, *(param->ctx));


        threshold = (size_t)(param->numElems/2);
        unique->SetSizes(localSize, (size_t)param->numElems); //give threshold as input


        //Use the same input and output buffers 	
        for(int j = 0; j < npasses; j++)
        {
            //For single input tests
            //Clear and reinitialize the input vector
            InitializeHostVector(&(param->mInputVals[0]), param->numElems, false, param->joinSize);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize(param->numElems);

            unique->RunCPUReference(cpuTime, param->mInputVals[0], param->numElems);

            if(param->verbose)
                Println("CPU : [%f secs]", cpuTime);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(unique->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
            unique->Run(*param);			

            //To avoid memory leaks on multiple passes, explicitly free intermediate mem buffers	
            unique->FreeDevBuffers();	

            
        // update progress bar
        pb->addItersDone();
        if (!param->verbose && !param->quiet)
             pb->Show(stdout);
        
        }//End npasses
        
        //Free device buffers
        DeallocateDevBuffer(param->memInput[0]);
        DeallocateDevBuffer(param->memOutput);	

        free(unique);

    }
    else if(param->testName == "Project")
    {

        ProjectApp *project;
        project = new ProjectApp();

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Output
        AllocateDevBuffer(&param->memOutput, (double)(param->devBufferSz), 1, *(param->ctx));


        threshold = (size_t)(param->numElems/2);
        project->SetSizes(localSize, (size_t)param->numElems); //give threshold as input


        //Use the same input and output buffers 	
        for(int j = 0; j < npasses; j++)
        {
            //For single input tests
            //Clear and reinitialize the input vector
            InitializeHostVector(&(param->mInputVals[0]), param->numElems, false, param->joinSize);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize(param->numElems);

            project->RunCPUReference(cpuTime, param->mInputVals[0], param->numElems);

            if(param->verbose)
                Println("CPU : [%f secs]", cpuTime);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(project->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
            project->Run(*param);			

            //To avoid memory leaks on multiple passes, explicitly free intermediate mem buffers	
        //    project->FreeDevBuffers();	

            
        // update progress bar
        pb->addItersDone();
        if (!param->verbose && !param->quiet)
             pb->Show(stdout);
        
        }//End npasses
        
        //Free device buffers
        DeallocateDevBuffer(param->memInput[0]);
        DeallocateDevBuffer(param->memOutput);	

        free(project);

    }
    else if(param->testName == "A")
    {
        //Perform 3 Select operations
        int numChainedOps = 3;

        SelectionApp *selection[numChainedOps];
        size_t selectOutSz[numChainedOps];

        for(int i = 0; i < numChainedOps; i++)
        {
            selection[i] = new SelectionApp();
        }

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Create input and output buffers on the host

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Output
        AllocateDevBuffer(&param->memOutput, (double)(param->devBufferSz), 1, *(param->ctx));

        threshold = (size_t)(param->numElems/2);
        //Each select iteration operates over slightly smaller datasets
        selection[0]->SetSizes(localSize, (size_t)param->numElems, threshold); //give threshold as input
        selection[1]->SetSizes(localSize, (size_t)param->numElems, threshold/2); 
        selection[2]->SetSizes(localSize, (size_t)param->numElems, param->numElems); 
        //Use the same input and output buffers 	
        for(int j = 0; j < npasses; j++)
        {
            //For single input tests
            //Clear and reinitialize the input vector
            InitializeHostVector(&(param->mInputVals[0]), param->numElems, false, param->joinSize);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize(param->numElems);

            selectOutSz[0] = selection[0]->RunCPUReference(cpuTime, param->mInputVals[0], param->numElems);

            if(param->verbose)
                Println("CPU 1: [%f secs]", cpuTime);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(selection[0]->SetBuffers(1, *param), CL_SUCCESS, "setBuffers failed.");
            selection[0]->Run(1, *param);			

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            selection[0]->FreeDevBuffers();

            //-----------------Select 2-------------------
            //Select 2 - use output of Select 1 as input
            param->mInputVals[0] = param->mOutputVals;
            
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize(param->numElems/2);

            selectOutSz[1] = selection[1]->RunCPUReference(cpuTime, param->mInputVals[0], selectOutSz[0]);

            if(param->verbose)
                Println("CPU 2: [%f secs]", cpuTime);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(selection[1]->SetBuffers(1, *param), CL_SUCCESS, "setBuffers failed.");
            selection[1]->Run(1, *param);			

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            selection[1]->FreeDevBuffers();
            
            //-----------------Select 3-------------------
            //Select 3 - use output of Select 2 as input
            param->mInputVals[0] = param->mOutputVals;
            
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            //TODO - why is this value bigger?
            param->mOutputVals.resize(param->numElems);

            selectOutSz[2] = selection[2]->RunCPUReference(cpuTime, param->mInputVals[0], selectOutSz[0]);

            if(param->verbose)
                Println("CPU 2: [%f secs]", cpuTime);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(selection[2]->SetBuffers(1, *param), CL_SUCCESS, "setBuffers failed.");
            selection[2]->Run(1, *param);			

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            selection[2]->FreeDevBuffers();

            // update progress bar
            pb->addItersDone();
            if (!param->verbose && !param->quiet)
                pb->Show(stdout);

        }//End npasses


        //Free both buffers
        DeallocateDevBuffer(param->memInput[0]);
        DeallocateDevBuffer(param->memOutput);	

        for(int i = 0; i < numChainedOps; i++)
            free(selection[i]);

    }
    else if(param->testName == "B")
    {

        //Perform 2 Join operations
        int numChainedOps = 2;

        InnerJoinApp *join[numChainedOps];
        size_t joinOutSz[numChainedOps];

        for(int i = 0; i < numChainedOps; i++)
        {
            join[i] = new InnerJoinApp();
        }

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Create input and output buffers on the host

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Input 2 
        AllocateDevBuffer(&param->memInput[1], (double)(param->devBufferSz), 0, *(param->ctx));
        //Output - (2xinput) * 2
        //AllocateDevBuffer(&param->memOutput, (double)(param->devBufferSz*4), 1, *(param->ctx));

        size_t joinLeftNumElem;
        bool isFirstFlag;

        //Use the same input and output buffers 	
        for(int j = 0; j < npasses; j++)
        {
            if(param->verbose)
                cout<<endl<<"Pass "<<j<<" of "<<npasses<<endl;

            InitializeHostVector(&(param->mInputVals[0]), param->numElems, true, param->joinSize);
            InitializeHostVector(&(param->mInputVals[1]), param->numElems, true, param->joinSize);
            InitializeHostVector(&(param->mInputVals[2]), param->numElems, true, param->joinSize);
          
            //The first join operation is handled differently
            isFirstFlag = true; 
            
            for(int n = 0; n < numChainedOps; n++)
            {
                //The input size for the left-hand input 
                //is dependent on the previous iteration's output
                if(n == 0)
                    joinLeftNumElem = param->numElems;
                else
                {
                    joinLeftNumElem = joinOutSz[n-1];
                    isFirstFlag = false;
                
                }

                join[n]->SetSizes(localSize, (size_t)param->numElems, (size_t)param->numElems, isFirstFlag); //give threshold as input
                //Clear the output vector
                ResetHostVector(&param->mOutputVals);

                joinOutSz[n] = join[n]->RunCPUReference(cpuTime, param->mInputVals[0], param->mInputVals[1], joinLeftNumElem, param->numElems, true);
                param->cpuOutputSzBytes = joinOutSz[n]*sizeof(Tuple);
                
                if(param->verbose)
                    Println("CPU : [%f secs]; Output Size: [%d]; OutputBytes: [%d]", cpuTime, joinOutSz[n], param->cpuOutputSzBytes);
           
                if(joinOutSz[n] == 0)
                {
                    if(param->verbose)
                        cout<<"This join does not have any valid outputs - skipping accelerated version!"<<endl;

                    break;
                }

                //Resize the output vector to hold new outputs
                param->mOutputVals.resize(param->cpuOutputSzBytes+sizeof(Tuple));
            
                //TODO - this slows down overall execution but allows larger tests
                AllocateDevBuffer(&param->memOutput, (double)(param->cpuOutputSzBytes+sizeof(Tuple)), 1, *(param->ctx));

                //Note that SetBuffers creates buffers and copies data to the device
                CheckError(join[n]->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
                join[n]->Run(*param);			

                //Move output to be the left-hand input
                //and the next input vector to be the right-hand input
                param->mInputVals[0] = param->mOutputVals;
                param->mInputVals[1] = param->mInputVals[n+2];
                
                //To avoid memory leans on multiple passes, explicitly free mem buffers	
                join[n]->FreeDevBuffers();
                //TODO - this slows down overall execution but allows larger tests
                DeallocateDevBuffer(param->memOutput);	
            }

        }//end npasses
        
        //Free all device buffers
        DeallocateDevBuffer(param->memInput[0]);
        DeallocateDevBuffer(param->memInput[1]);

        for(int i = 0; i < numChainedOps; i++)
            free(join[i]);

    }//End B
    else if(param->testName == "C")
    {
	//Perform 3 select operations, join the outputs using a join chain
	//and then perform a select operation on the result
       
	//---------Select setup-------------------- 
        int numSelectOps = 3;

        SelectionApp *selection[numSelectOps];
        size_t selectOutSz[numSelectOps];

        for(int i = 0; i < numSelectOps; i++)
        {
            selection[i] = new SelectionApp();
        }

        //Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        threshold = (size_t)(param->numElems/2);
        //Each select iteration operates over slightly smaller datasets
        selection[0]->SetSizes(localSize, (size_t)param->numElems, threshold); //give threshold as input
        selection[1]->SetSizes(localSize, (size_t)param->numElems, threshold/2); 
        selection[2]->SetSizes(localSize, (size_t)param->numElems, param->numElems); 
	
	//-----------Join setup---------------------
	//Perform 2 Join operations
        int numChainedJoins = 2;

        InnerJoinApp *join[numChainedJoins];
        size_t joinOutSz[numChainedJoins];

        for(int i = 0; i < numChainedJoins; i++)
        {
            join[i] = new InnerJoinApp();
        }
        
	size_t joinLeftNumElem;
        bool isFirstFlag;

	//----------Project setup-------------------
	size_t projectSz;	
        ProjectApp *project;
        project = new ProjectApp();

	//-----------------------------------------        

	//Handle case where input is smaller than workgroup size
        if(localSize > param->numElems)
            localSize = param->numElems;

        //Create input and output buffers on the host

        //Input 1 
        AllocateDevBuffer(&param->memInput[0], (double)(param->devBufferSz), 0, *(param->ctx));
        //Input 2 
        AllocateDevBuffer(&param->memInput[1], (double)(param->devBufferSz), 0, *(param->ctx));
        //Input 3
	AllocateDevBuffer(&param->memInput[2], (double)(param->devBufferSz), 0, *(param->ctx));

        //Output - note that this needs to be reallocated for joins (which may be bigger)
        AllocateDevBuffer(&param->memOutput, (double)(param->devBufferSz), 1, *(param->ctx));
        
	//Use the same device input and output buffers except for Joins
        for(int j = 0; j < npasses; j++)
        {

            if(param->verbose)
                cout<<endl<<"Pass "<<j<<" of "<<npasses<<endl;

            InitializeHostVector(&(param->mInputVals[0]), param->numElems, true, param->joinSize);
            InitializeHostVector(&(param->mInputVals[1]), param->numElems, true, param->joinSize);
            InitializeHostVector(&(param->mInputVals[2]), param->numElems, true, param->joinSize);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            param->mOutputVals.resize(param->numElems);

	    //=====================3 Select operations==========================
          
            selectOutSz[0] = selection[0]->RunCPUReference(cpuTime, param->mInputVals[0], param->numElems);
            
	    if(param->verbose)
                Println("CPU Select 1: [%f secs], OutputSz: %d", cpuTime, selectOutSz[0]);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(selection[0]->SetBuffers(1, *param), CL_SUCCESS, "setBuffers failed.");
            selection[0]->Run(1, *param);			
            
	    //Save the output in a new input vector
	    param->mInputVals[3] = param->mOutputVals;

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            selection[0]->FreeDevBuffers();
            
	    //-----------------Select 2-------------------
            //Select 2 - use next input
            param->mInputVals[0] = param->mInputVals[1];
            
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize(param->numElems/2);

            selectOutSz[1] = selection[1]->RunCPUReference(cpuTime, param->mInputVals[0], selectOutSz[0]);

            if(param->verbose)
                Println("CPU Select 1: [%f secs], OutputSz: %d", cpuTime, selectOutSz[1]);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(selection[1]->SetBuffers(1, *param), CL_SUCCESS, "setBuffers failed.");
            selection[1]->Run(1, *param);			
	    
	    //Save the output in a new input vector
	    param->mInputVals[4] = param->mOutputVals;

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            selection[1]->FreeDevBuffers();
	    
	    //-----------------Select 3-------------------
            //Select 3 - use next input
            param->mInputVals[0] = param->mInputVals[2];
            
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);
            //But resize the output vector to hold new outputs
            param->mOutputVals.resize(param->numElems);

            selectOutSz[2] = selection[2]->RunCPUReference(cpuTime, param->mInputVals[0], selectOutSz[0]);

            if(param->verbose)
                Println("CPU Select 2: [%f secs], OutputSz: %d", cpuTime, selectOutSz[2]);

            //Note that SetBuffers creates buffers and copies data to the device
            CheckError(selection[2]->SetBuffers(1, *param), CL_SUCCESS, "setBuffers failed.");
            selection[2]->Run(1, *param);			
	    
	    //Save the output in a new input vector
	    param->mInputVals[5] = param->mOutputVals;

            //To avoid memory leaks on multiple passes, explicitly free mem buffers	
            selection[2]->FreeDevBuffers();
            

	   //======================2 Join operations============================
	    //The first join operation is handled differently
            isFirstFlag = true; 
            
            
            param->mInputVals[0] = param->mInputVals[3];
            param->mInputVals[1] = param->mInputVals[4];

            join[0]->SetSizes(localSize, (size_t)selectOutSz[0], (size_t)selectOutSz[1], isFirstFlag);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);

            joinOutSz[0] = join[0]->RunCPUReference(cpuTime, param->mInputVals[0], param->mInputVals[1], selectOutSz[0], selectOutSz[1], true);
            param->cpuOutputSzBytes = joinOutSz[0]*sizeof(Tuple);
                
                if(param->verbose)
                    Println("CPU J1: [%f secs]; Output Size: [%d]; OutputBytes: [%d]", cpuTime, joinOutSz[0], param->cpuOutputSzBytes);
           
                if(joinOutSz[0] == 0)
                {
                    if(param->verbose)
                        cout<<"This join does not have any valid outputs - skipping accelerated version!"<<endl;
			//Set different inputs to try and do future operations on a non-empty set
                	param->mInputVals[0] = param->mInputVals[3];
                	param->mInputVals[1] = param->mInputVals[5];
		    	joinOutSz[0] = param->mInputVals[3].size();
		}
		else
		{
                //Resize the output vector to hold new outputs
                param->mOutputVals.resize(param->cpuOutputSzBytes+sizeof(Tuple));
            
                //TODO - this slows down overall execution but allows larger tests
                AllocateDevBuffer(&param->memOutput, (double)(param->cpuOutputSzBytes+sizeof(Tuple)), 1, *(param->ctx));

                //Note that SetBuffers creates buffers and copies data to the device
                CheckError(join[0]->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
                join[0]->Run(*param);			
                
		//Move output to be the left-hand input for next operation
                param->mInputVals[0] = param->mOutputVals;
                param->mInputVals[1] = param->mInputVals[5];
                
                //To avoid memory leans on multiple passes, explicitly free mem buffers	
                join[0]->FreeDevBuffers();
		}//end else

                //TODO - this slows down overall execution but allows larger tests
                DeallocateDevBuffer(param->memOutput);	

		//-----------------Join 2-----------------------------
	    
	    //The first join operation is handled differently
            isFirstFlag = false; 

            join[1]->SetSizes(localSize, (size_t)joinOutSz[0], (size_t)selectOutSz[2], isFirstFlag);
            //Clear the output vector
            ResetHostVector(&param->mOutputVals);

            joinOutSz[1] = join[1]->RunCPUReference(cpuTime, param->mInputVals[0], param->mInputVals[1], joinOutSz[0], selectOutSz[2], true);
            param->cpuOutputSzBytes = joinOutSz[1]*sizeof(Tuple);
                
                if(param->verbose)
                    Println("CPU J2: [%f secs]; Output Size: [%d]; OutputBytes: [%d]", cpuTime, joinOutSz[1], param->cpuOutputSzBytes);
           
                if(joinOutSz[1] == 0)
                {
                    if(param->verbose)
                        cout<<"This join does not have any valid outputs - skipping accelerated version!"<<endl;
                    param->mInputVals[0] = param->mInputVals[3];
		    joinOutSz[1] = param->mInputVals[3].size();
            	    param->cpuOutputSzBytes = joinOutSz[1]*sizeof(Tuple);
		}
		else
		{

                //Resize the output vector to hold new outputs
                param->mOutputVals.resize(param->cpuOutputSzBytes+sizeof(Tuple));
            
                //TODO - this slows down overall execution but allows larger tests
                AllocateDevBuffer(&param->memOutput, (double)(param->cpuOutputSzBytes+sizeof(Tuple)), 1, *(param->ctx));

                //Note that SetBuffers creates buffers and copies data to the device
                CheckError(join[1]->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
                join[1]->Run(*param);			
                
		//Move output to be the left-hand input for next operation
                param->mInputVals[0] = param->mOutputVals;
                
                //To avoid memory leans on multiple passes, explicitly free mem buffers	
                join[1]->FreeDevBuffers();
		}
		//===================1 Project operation==============================
			
        	AllocateDevBuffer(&param->memOutput, (double)(param->cpuOutputSzBytes), 1, *(param->ctx));

        	//Handle case where input is smaller than workgroup size
        	if(localSize > joinOutSz[1])
            		localSize = joinOutSz[1];

	
        	project->SetSizes(localSize, (size_t)joinOutSz[1]); //give threshold as input

            	//Clear the output vector
            	ResetHostVector(&param->mOutputVals);
            	//But resize the output vector to hold new outputs
            	param->mOutputVals.resize(joinOutSz[1]);
            
		projectSz = project->RunCPUReference(cpuTime, param->mInputVals[0], joinOutSz[1]);
                if(param->verbose)
                    Println("CPU Project: [%f secs]; Output Size: [%d]", cpuTime, projectSz);

            	CheckError(project->SetBuffers(*param), CL_SUCCESS, "setBuffers failed.");
            	project->Run(*param);			

                //TODO - this slows down overall execution but allows larger tests
                DeallocateDevBuffer(param->memOutput);	
        
		// update progress bar
        	pb->addItersDone();
        	if (!param->verbose && !param->quiet)
             		pb->Show(stdout);
        }//end npasses
        
        //Free all device buffers and application pointers
        for(int i = 0; i < numSelectOps; i++)
	{
        	DeallocateDevBuffer(param->memInput[i]);
            	free(selection[i]);
	}
        
	for(int i = 0; i < numChainedJoins; i++)
            free(join[i]);

	free(project);

    }//End C
    else
        cout<<"Please enter a test name"<<endl;

}


void Println(const string format, ...)
{
    char buffer[16384];
    va_list args;

    va_start(args, format);
    vsnprintf(buffer, sizeof(char)*16384, format.c_str(), args);
    std::cout << buffer << std::endl;
    va_end(args);
}

void CheckError(const int actual, const int reference, const string msg)
{
    if (actual != reference)
    {
        Println(msg);
        exit(EXIT_FAILURE);
    }
}

//reads contents of a file into a string.
string ReadFileToString(string fileName)
{
    ifstream stream(fileName.c_str(), ifstream::in);

    if (!stream.good())
    {
        Println("Failed to open file - %s", fileName.data());
        exit(0);
    }

    return string(std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>());
}

int GetDeviceSettings(BmkParams* bmk){

    cl_int err;

    int numSMs = getMaxComputeUnits(*(bmk->dev));
    //Get the OpenCL limit on allocations
    clGetDeviceInfo(*(bmk->dev), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_long), &(bmk->maxClAllocBytes), NULL);

    cout<<"Max allocation size for this device is "<<bmk->maxClAllocBytes<<" B  and "<<((double)(bmk->maxClAllocBytes)/(1024.0*1024.0))<<" MB"<<endl;

    //Get the type of device - CPU, GPU, etc. and vendor
    clGetDeviceInfo(*(bmk->dev), CL_DEVICE_TYPE, sizeof(cl_device_type), &(bmk->devType), NULL);

    clGetDeviceInfo(*(bmk->dev), CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &(bmk->vendorId), NULL);

    clGetDeviceInfo(*(bmk->dev), CL_DEVICE_VENDOR, sizeof(char)*128, &(bmk->dev_vendor_id), NULL);


    //Copied over from common code since there isn't an external function
     size_t nBytesNeeded = 0;
	char dev_vendor_id[128];
        err = clGetDeviceInfo( bmk->dev[0],
                                CL_DEVICE_NAME,
                                0,
                                NULL,
                                &nBytesNeeded );
        CL_CHECK_ERROR(err);
        char* devName = new char[nBytesNeeded+1];
        err = clGetDeviceInfo( bmk->dev[0],
                                CL_DEVICE_NAME,
                                nBytesNeeded+1,
                                devName,
                                NULL );

     bmk->devName = devName;

    //Based on the type of device, set the workgroup size and vector size

    //AMD:
    //Workgroup - 256
    //Max 1 alloc: 128 MB
    //Vector size: 64

    /*for(int pl=0; pl!=platforms.size(); pl++){
      platform = platforms[pl];
      std::string name = platform.getInfo<CL_PLATFORM_NAME>();
      if(platformArg == "AMD"){
      if( platforms[pl].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing"){	
      break;
      }
      }
      else if(platformArg == "Intel"){
      if( platforms[pl].getInfo<CL_PLATFORM_NAME>() == "Intel(R) OpenCL"){	
      break;
      }
      }
      else if(platformArg == "Nvidia"){
      if( platforms[pl].getInfo<CL_PLATFORM_NAME>() == "NVIDIA CUDA"){	
      break;
      }
      }
      }

      if(platformArg == "AMD"){
      if(platform.getInfo<CL_PLATFORM_NAME>() != "AMD Accelerated Parallel Processing"){
      Println("Required platform not found.");
      return EXIT_FAILURE;
      }
      }
      else if(platformArg == "Intel"){
      if(platform.getInfo<CL_PLATFORM_NAME>() != "Intel(R) OpenCL"){
      Println("Required platform not found.");
      return EXIT_FAILURE;
      }
      }
      else if(platformArg == "Nvidia"){
      if(platform.getInfo<CL_PLATFORM_NAME>() != "NVIDIA CUDA"){
      Println("Required platform not found.");
      return EXIT_FAILURE;
      }
      }*/

    return CL_SUCCESS;
}


//create and build kernel program.
int SetProgram(string fileName, cl_context ctx, cl_device_id dev) 
{
    string sourceStr;
    cl_int err;
    string clFileName = fileName;

    // OpenCL compiler options -- default is to enable
    // all optimizations
    //string opts = "-cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only";

    //***USE THIS STRING TO DEBUG INTO THE OPENCL KERNELS WITH INTEL COMPILERS***
    string opts = "-g -s \"" + clFileName + "\" -cl-opt-disable";
    //string opts = "-cl-opt-disable";

    //Right now (Dec. 2014) Intel OpenCL implementations have a weird issue with their vectorizing
    //component which manifests as a segfault. For this reason, optimizations are turned off.
    //string opts = "-cl-opt-disable";

    //***USE THIS STRING TO DEBUG INTO THE OPENCL KERNELS WITH AMD COMPILERS***
    //string opts = "-g -O0";

    //read the kernel source file.
    sourceStr = ReadFileToString(fileName);

    //create program after converting from string to char*
    const char* progSource[] = {sourceStr.c_str()};
    program = clCreateProgramWithSource(ctx, 1, progSource, NULL, &err);
    CL_CHECK_ERROR(err);

    //build program and specify our build options
    cl_int ret_val = clBuildProgram(program, 1, &dev, opts.c_str(), NULL, NULL);
    //cl_int ret_val = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    //------------------Check build errors using online sample
    // avoid abort due to CL_BILD_PROGRAM_FAILURE

    if (ret_val != CL_SUCCESS && ret_val != CL_BUILD_PROGRAM_FAILURE)
        CL_CHECK_ERROR(ret_val);

    cl_build_status build_status;

    CL_CHECK_ERROR(clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL));

    if (build_status == CL_SUCCESS)

        return CL_SUCCESS;


    char *build_log;

    size_t ret_val_size;

    CL_CHECK_ERROR(clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));

    build_log = new char[ret_val_size+1];

    CL_CHECK_ERROR(clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

    // to be careful, terminate with \0

    // there's no information in the reference whether the string is 0 terminated or not

    build_log[ret_val_size] = '\0';

    std::cout << "BUILD LOG: '" << "'" << std::endl;

    std::cout << build_log << std::endl;

    delete[] build_log;


    CL_CHECK_ERROR(err);

    return CL_SUCCESS;
}

//Generates a random value within specified bounds.
valType GetUrandom(valType vmax) 
{

    return (static_cast<valType>(rand())) % vmax;
}

//Determine what index range to use for this device
void SetTestRange(BmkParams* param, int* minIdx, int* maxIdx)
{ 
    unsigned long long tupleSz = sizeof(Tuple);
    unsigned long long inputElemSizes[27] = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
        32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,
        67108864,134217728};
    int cnt = 0;

    //tupleSz * inputElemSizes[n]
    unsigned long long inputSize;

    if(param->numElems != 0)
    {
        inputSize = tupleSz*inputElemSizes[cnt];

        while(inputElemSizes[cnt] < param->numElems)
        {
            cnt++; 	
            inputSize = tupleSz*inputElemSizes[cnt];

            if(param->verbose)
                cout<<"Elems"<<inputElemSizes[cnt]<<" Input size (B) "<<inputSize<<" numElem"<<param->numElems<<endl;
            if(inputSize >= param->maxClAllocBytes)
                break;
        }

        if(inputElemSizes[cnt] != param->numElems)
            cout<<"Input elements, "<<param->numElems
                <<" doesn't quite match valid size, "
                <<inputElemSizes[cnt]<<" so changing input size"<<endl; 

        *minIdx = cnt;
        *maxIdx = cnt;
    }
    else
    {
        do
        {
            inputSize = tupleSz*inputElemSizes[cnt];
            cnt++; 	
            cout<<"Input size"<<inputSize<<endl;
        }
        while(inputSize < param->maxClAllocBytes);

        *minIdx = 7;
        *maxIdx = cnt;
    }

    //Size the input and output buffers to hold the largest test input	
    param->devBufferSz=(inputElemSizes[cnt]*tupleSz);

    if(param->verbose)
    {
        cout<<"Max Index is "<<*maxIdx<<" and max buffer size is "<<param->devBufferSz<<endl;
    }


}

//Helper function to find the correct work group size for a kernel
size_t FindKernelWorkGroupSize(cl_kernel krn, BmkParams param, size_t globalSz, size_t localSz)
{
    size_t krnWorkgroupMax;
    size_t devWorkgroupMax;
    size_t krnWorkgroupMultiple;
    size_t selWgSz;
    size_t numGroups = globalSz / localSz;
    cl_int err;

    err = clGetDeviceInfo(*(param.dev), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &devWorkgroupMax, NULL);

    //Find the max workgroup size and the preferred multiple for this device
    err = clGetKernelWorkGroupInfo(krn, *(param.dev), CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t), &krnWorkgroupMax, NULL);
    err = clGetKernelWorkGroupInfo(krn, *(param.dev), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            sizeof(size_t), &krnWorkgroupMultiple, NULL);



    if((globalSz % krnWorkgroupMultiple) == 0)
        selWgSz = krnWorkgroupMultiple;
    else
        selWgSz = localSz;

    /*if(krnWorkgroupMax < selWgSz)
      selWgSz = krnWorkgroupMax;

      if(devWorkgroupMax < selWgSz)
      selWgSz = devWorkgroupMax;*/

    if(param.verbose)
    {
        cout<<" Kernel max workgroup size "<<krnWorkgroupMax<<" preferred size "<<krnWorkgroupMultiple<<" global/local sz: "<<globalSz<<"/"<<localSz<<endl;
        cout<<"Dev max "<<devWorkgroupMax<<endl;
        cout<<"Selected wg size of "<<selWgSz<<endl;
    }

    return selWgSz;
}


void InitializeHostVector(vector<Tuple>* hVect, size_t mNumElements, bool isJoin, int joinSize)
{

    //Each time we initialize a host vector, we clear the 
    //previous contents
    hVect->clear();

    //Find the maximum random value for this particular datatype
    //We will add 1 to the randomized number in order to avoid too many 
    //0 values as keys or values
    valType mod;
    
    //Check for ushort vs. uint values for the max value
    if(sizeof(valType) == 8)
        mod  = USHRT_MAX-1;
    else if(isJoin) //Use a small modifier so joins are not on 0-element arrays
    {
	switch(joinSize)
	{
		//Highest join intensity - output is reasonably larger than
		//the sum of input tuple arrays
		case 4:
			mod = (valType)(mNumElements/2);
		  break;
		case 3:
			mod = (valType)mNumElements;
		  break;
		case 2:
			mod = (valType)(mNumElements*2);
		  break;
		case 1: //expanded range of keys makes joins less likely
			mod = (valType)(mNumElements*4444);
		  break;
	}

/*        if(mNumElements < 32768)
	{
            //mod = (valType)32768;
            mod = (valType)mNumElements;
	}
        else
	{
            //mod = (valType)32768;
	    mod = 1048576;
    	}*/
    }
    else
    {
        mod = UINT_MAX-1;
    }

    for(int sz=0; sz<mNumElements; sz++)
    {
         Tuple tpl;

            //Pick a non-zero value for the key
            tpl.key = GetUrandom(mod) + 1;
            //key2 is rarely used except for Product
            //tpl.key2 = 0;
            tpl.valArray[0] = GetUrandom(mod) + 1;
            tpl.valArray[1] = 0;

            hVect->push_back(tpl);
    }

    //Some benchmarks like innerJoin require sorted input
    if(isJoin)
        std::sort(hVect->begin(),hVect->end()); 

}

//Reset a host-based vector
void ResetHostVector(vector<Tuple>* hVect)
{
    hVect->clear();
}

//Allocate an input or output buffer on the device that gets reused
void AllocateDevBuffer(cl_mem* mBuf, double sizeBytes, int rwFlag, cl_context ctx)
{
    cl_int err;

    if(sizeBytes == 0)
    {
	cout<<"No output for join operation! Please use a larger join size to increase size of output"<<endl;
	exit(1);
    }
	

    //cout<<"Allocating buffer of size "<<sizeBytes<<" B and "<<((double)sizeBytes/(1024.0*1024.0))<<" MB"<<endl<<endl;

    if(rwFlag == 0) //Read-only buffer
    {
        *mBuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeBytes, NULL, &err);
        CL_CHECK_ERROR(err);
    }
    else //RW buffer for outputs
    {
        *mBuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
        CL_CHECK_ERROR(err);
    }

}

//Deallocate an input or output buffer on the device
void DeallocateDevBuffer(cl_mem mBuf)
{
    cl_int err;

    err = clReleaseMemObject(mBuf);
    CL_CHECK_ERROR(err);
}

void AddTimeResultToDB(BmkParams param, double dataInNs, size_t dataSz, string rsltNm)
{ 

    char sizeStr[256];
    unsigned long long dataSizekB = dataSz/1024;
    sprintf(sizeStr, "%7llukB",dataSizekB);
    double nsToSec = 1.e-9;
    double dataInSec = dataInNs*nsToSec;;

    (param.resultDB)->AddResult(rsltNm.c_str(), sizeStr,"s", dataInSec);
}

//Copy out sizeBytes bytes from device buffer, devBuf, to a
//pointer array and print out its contents
void DebugDevMem(BmkParams param, cl_mem devBuf, size_t numElems, bool isTuple, string id)
{
        unsigned int sizeBytes;
	
	cout<<endl<<"------Debug: "<<id<<"------"<<endl;

	//Just print out numeric values
	if(!isTuple)
	{	
		sizeBytes = numElems*sizeof(unsigned int);
	        unsigned int debugHostBuf[numElems+1];

        	cl_int err = clEnqueueReadBuffer(*(param.queue), devBuf, CL_TRUE, 0, sizeBytes, debugHostBuf, 0, NULL, NULL);
        	CL_CHECK_ERROR(err);
		cout<<"Host address: "<<debugHostBuf<<", device: "<<devBuf<<endl;

        	for(int i = 0; i < numElems; i++)
                	cout<<"["<<i<<"]: "<<debugHostBuf[i]<<" "<<endl;


	}
	else //print out tuples
	{
		sizeBytes = (numElems)*sizeof(Tuple);
	        Tuple debugHostBuf[numElems+1];

        	cl_int err = clEnqueueReadBuffer(*(param.queue), devBuf, CL_TRUE, 0, sizeBytes, &debugHostBuf, 0, NULL, NULL);
        	CL_CHECK_ERROR(err);
		cout<<"Host address: "<<debugHostBuf<<", device: "<<devBuf<<endl;

        	for(int i = 0; i < numElems; i++)
                	cout<<"["<<i<<"]: "<<debugHostBuf[i].key<<", "<<debugHostBuf[i].valArray[0]<<", "<<debugHostBuf[i].valArray[1]<<endl;

	}
	
	cout<<endl<<"----------------------"<<endl;

}

int VerifyResults(BmkParams param, vector<Tuple> cpuRslt, vector<Tuple> accRslt)
{
	int diffCount = 0;

        for(int i = 0; i < cpuRslt.size(); i++)
	{
	    if ((cpuRslt[i].key != accRslt[i].key) && (cpuRslt[i].valArray[0] != accRslt[i].valArray[0]))
	    {
	      if(param.verbose && (diffCount < 128))
	      {
                cout<<"["<<i<<"]: cpu="<<cpuRslt[i].key<<", "<<cpuRslt[i].valArray[0]<<", "<<cpuRslt[i].valArray[1]<<endl;
                cout<<"["<<i<<"]: accel="<<accRslt[i].key<<", "<<accRslt[i].valArray[0]<<", "<<accRslt[i].valArray[1]<<endl;
	      }
      		diffCount++;
            }
  	}

	return diffCount;
}

void ZeroDevBuffer(BmkParams param, cl_mem devBuf, size_t sizeBytes)
{
    	unsigned int zeroPattern = 0;
	cl_int err;

	err = clEnqueueFillBuffer(*(param.queue), devBuf, &zeroPattern, sizeof(unsigned int), 0, (size_t)(sizeBytes), 0, NULL, NULL);
 	CL_CHECK_ERROR(err);

}
