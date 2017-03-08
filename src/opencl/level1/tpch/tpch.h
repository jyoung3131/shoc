#ifndef TPCH_DEF
#define TPCH_DEF

#define __CL_ENABLE_EXCEPTIONS
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
//NVIDIA CUDA Linux
#include <CL/opencl.h>

//C++ bindings aren't supported by SHOC
//#include <CL/cl.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
//#include <random>
#include <limits.h>
#include <algorithm>
#include <stdlib.h>
#include <assert.h>
#include <cstdarg>
#include "Event.h"
#include "Timer.h"
#include "ResultDatabase.h"
#include "ProgressBar.h"
//Includes error-checking code and functions to get max workgroup size
#include "support.h"
//#include <vexcl/vexcl.hpp>

using namespace std;

//Data type for input elements
//typedef cl_ushort valType;
typedef cl_uint valType;

extern const cl_platform_id       platform;			//OpenCL platform
extern cl_program program;

typedef union
{
	struct _tuple
	{
		valType key;
		valType valArray[3];
		bool operator<(const _tuple& rhs) const {
			if (key == rhs.key) {
				if (valArray[1] == 0) return valArray[0] < rhs.valArray[0];
				else {
					if (valArray[0] == rhs.valArray[0]) return valArray[1] < rhs.valArray[1];
					else return valArray[0] < rhs.valArray[0];
				}
			}
			else return key < rhs.key;
		}
	} tuple;
	struct _prod_tuple
	{
		valType keyArray[2];
		valType valArray[2];
		bool operator<(const _prod_tuple& rhs) const {
			if (keyArray[0] == rhs.keyArray[0]) {
				if (keyArray[1] == rhs.keyArray[1]) {
					if (valArray[0] == rhs.valArray[0]) return valArray[1] < rhs.valArray[1];
					else return valArray[0] < rhs.valArray[0];
				}
				else return keyArray[1] < rhs.keyArray[1];
			}
			else return keyArray[0] < rhs.keyArray[0];
		}
	} prod_tuple;

} Tuple;
bool join_comp(Tuple i, Tuple j);
/*
bool join_comp(Tuple i, Tuple j)
{
	if (i.tuple.key == j.tuple.key) {
		if (i.tuple.valArray[1] == 0) return i.tuple.valArray[0] < j.tuple.valArray[0];
		else {
			if (i.tuple.valArray[0] == j.tuple.valArray[0]) return i.tuple.valArray[1] < j.tuple.valArray[1];
			else return i.tuple.valArray[0] < j.tuple.valArray[0];
		}
	}
	else return i.tuple.key < j.tuple.key;
}
*/
/*typedef struct prodTuple
{
    valType key, key2;
    valType valArray[2];

}*/

//With cl_int, the size of this struct is 16 B; otherwise 8 B

/*typedef struct myTuple
{
    //Some operations like product support the combination of multiple keys
    //but adding an array would cause many code changes so is listed as TODO
    valType key;
    //TODO - do we actually need a 3 element valArray?
	valType valArray[3];
	bool operator<(const myTuple& rhs) const { 
		if(key<rhs.key) return true;
		else if(key>rhs.key) return false;
		else{ 
			if(valArray[1] == 0){
				if(valArray[0]<rhs.valArray[0]) return true; 
				else return false;
			}
			else{
				if(valArray[0]<rhs.valArray[0]) return true; 
				else if(valArray[0]>rhs.valArray[0]) return false; 
				else if(valArray[1]<rhs.valArray[1]) return true; 
				else return false;
			}
		}
	}
}Tuple;*/
/*extern cl_context        context;			//OpenCL context
extern vector<cl_device_id> devices;
extern vector<cl_device_id> device;			//Device to use
extern cl_command_queue   commandQueue;		//device command queue*/

/*extern cl::Platform       platform;			//OpenCL platform
extern cl::Context        context;			//OpenCL context
extern vector<cl::Device> devices;
extern vector<cl::Device> device;			//Device to use
extern cl::CommandQueue   commandQueue;		//device command queue
extern cl::Program		  program;
*/

//Struct that contains pointers to all the parameters set by SHOC
//Used to allow for simplicity in passing these values to functions
class BmkParams
{
	public:
	//pointers to the values defined elsewhere 
	cl_device_id *dev;
	cl_context *ctx;
	cl_command_queue *queue;
	cl_platform_id *platform;
	cl_program *program;
	ResultDatabase *resultDB;
	bool verbose;
	bool quiet;
	//number of input elements
	size_t numElems;
	//A value from 1-4 specifying the range of values for keys. Using a
	//smaller range decreases key variation and increases matches for inner
	//join operation
	int joinSize;
	//Specify the max size for an OpenCL allocation; typically this 
	//is on the order of 1/2 global memory
	cl_long maxClAllocBytes;
	cl_device_type devType;
	cl_uint vendorId;
	char dev_vendor_id[128];
	string devName;
	unsigned int project_col;	

	//Number of iterations to run a benchmark
	int nPasses;
	//Primitive name	
	string testName;
	//Device specific info

	//Constructor
	BmkParams(cl_device_id *Dev, cl_context *Ctx, cl_command_queue *Q, ResultDatabase *results, bool verboseFlag, bool quietFlag)
	{
		dev = Dev;
		ctx = Ctx;
		queue = Q;
		resultDB = results;
		verbose = verboseFlag;
		quiet = quietFlag;
		numElems = 256;
	}

	void SetNumElements(size_t elems){ numElems = elems;}
	
	//TODO - add functionality to set the min and max input
	//sizes based on OpenCL parameters
	void SetInputSizes(size_t inputSize, size_t maxInputSizeMB, size_t minInputSizeMB){}

	//Common input buffers for all benchmarks
	//-------------------------------------------------------
	//Multiple input buffers, single output buffer - devBufferSz is
	//the max buffer used in a series of tests
	cl_long devBufferSz;
    //The size of the CPU output in bytes - this can be used to "cheat" for Join
    //and product and only allocate the needed buffer size
	cl_long cpuOutputSzBytes;
	cl_mem memInput[4];
	cl_mem memOutput;

	//Input and output tuple vectors
	vector<Tuple> mInputVals[6];
	vector<Tuple> mOutputVals;


	//destructor only destroys the platform and program pointers
	//~BmkParams() {}
};

//default number of elements.
const size_t NUM_ELEMENTS = 64;

//default work-group size.
const size_t LOCAL_SIZE   = 1024;

const valType THRESHOLD = 5;

//Get settings for the device and add to benchmark struct
int GetDeviceSettings(BmkParams* bmk);

void Println(const string = "", ...);					//for printing stuff.
void CheckError(const int, const int, const string);	//for checking errors and exits program on error.
valType GetUrandom(valType maxVal);
int SetProgram(string fileName, cl_context ctx, cl_device_id dev); 
void RunTest(BmkParams* param, int npasses, ProgressBar* pb); 
void SetTestRange(BmkParams* param, int* minIdx, int* maxIdx); 

//Helper function to find the correct work group size for a kernel
size_t FindKernelWorkGroupSize(cl_kernel krn, BmkParams param, size_t globalSz, size_t localSz);

//Helper functions for memory

//Perform the initialization of each input vector
void InitializeHostVector(vector<Tuple>* hVect, size_t mNumElements, bool isJoin, int joinSize);
//Clear a host vector explicitly)
void ResetHostVector(vector<Tuple>* hVect);

//Allocate an input or output buffer that gets reused
void AllocateDevBuffer(cl_mem* mBuf, double sizeBytes, int rwFlag, cl_context ctx);
//Deallocate an input or output buffer
void DeallocateDevBuffer(cl_mem mBuf);

//Common function to save results to DB
void AddTimeResultToDB(BmkParams param, double dataInNs, size_t dataSz, string rsltNm);

//Copy out sizeBytes bytes from device buffer, devBuf, to a
//pointer array and print out its contents
void DebugDevMem(BmkParams param, cl_mem devBuf, size_t numElems, bool isTuple, string id);

int VerifyResults(BmkParams param, vector<Tuple> cpuRslt, vector<Tuple> accRslt);

//Zero out a device buffer as OpenCL doesn't have a default function to do this
void ZeroDevBuffer(BmkParams param, cl_mem devBuf, size_t sizeBytes);

#endif
