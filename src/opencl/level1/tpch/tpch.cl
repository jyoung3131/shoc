#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define CTAS 256
#define THREADS 256
#define CACHE_SIZE (1 << 10)

typedef float4 cltype;
typedef float type;
typedef unsigned int VAL_TYPE;

typedef struct myTuple
{
    VAL_TYPE key;
	VAL_TYPE val_array[3];
}Tuple;

typedef struct _prod_tuple
{
	VAL_TYPE key_array[2];
	VAL_TYPE val_array[2];
}prod_tuple;

__kernel void Addition(__global cltype* input1, const unsigned int input2, __global cltype* output, const int data_size){


	int global_size = get_global_size(0);  

	int id = get_global_id(0);

	int ceil_data_size = (data_size+3) /4;

	int count = 1;
	uint stride = global_size;
	if(ceil_data_size > global_size){
		count = ( (ceil_data_size) + global_size-1 )/global_size;  
	}
	for( int n=0; n < count; n++, id += stride ) {
	
		if(id < ceil_data_size){
			type temp1 = input1[id].x;
			temp1 += input2;
			output[id].x = temp1;
			temp1 = input1[id].y;
			temp1 += input2;
			output[id].y = temp1;
			temp1 = input1[id].z;
			temp1 += input2;
			output[id].z = temp1;
			temp1 = input1[id].w;
			temp1 += input2;
			output[id].w = temp1;

		}
	}
}


__kernel void Subtraction(__global cltype* input1, const unsigned int input2, __global cltype* output, const int data_size){

	int global_size = get_global_size(0);  

	int id = get_global_id(0);

	int ceil_data_size = (data_size+3) /4;

	int count = 1;
	uint stride = global_size;
	if(ceil_data_size > global_size){
		count = ( (ceil_data_size) + global_size-1 )/global_size;  
	}
	for( int n=0; n < count; n++, id += stride ) {
	
		if(id < ceil_data_size){
			type temp1 = input1[id].x;
			temp1 -= input2;
			output[id].x = temp1;
			temp1 = input1[id].y;
			temp1 -= input2;
			output[id].y = temp1;
			temp1 = input1[id].z;
			temp1 -= input2;
			output[id].z = temp1;
			temp1 = input1[id].w;
			temp1 -= input2;
			output[id].w = temp1;

		}
	}
}

__kernel void Multiplication(__global type* input1, __global type* input2, __global type* output, const int data_size){

	int global_size = get_global_size(0);  

	int id = get_global_id(0);

	int ceil_data_size = data_size;

	int count = 1;
	uint stride = global_size;
	if(ceil_data_size > global_size){
		count = ( (ceil_data_size) + global_size-1 )/global_size;  
	}
	for( int n=0; n < count; n++, id += stride ) {
	
		if(id < ceil_data_size){
			type temp1 = input1[id];
			type temp2 = input2[id];
			output[id] = temp1*temp2;
		}
	}
}

//Column 0 for key, 1 for data
__kernel void Project(__global Tuple* input, __global VAL_TYPE* output, const unsigned int num_elements, const unsigned int column)
{  	
	uint global_size = get_global_size(0);  
	uint global_id = get_global_id(0);
	uint local_id = get_local_id(0);
	uint local_size = get_local_size(0);
	uint group_id = get_group_id(0);
	uint groups = get_num_groups(0);

	const unsigned int partitions = groups;
	unsigned int chunk_size = (num_elements + partitions - 1) / partitions;
	unsigned int partition_size = (group_id != global_size - 1) ? chunk_size : (num_elements - group_id * chunk_size);

	unsigned int begin = chunk_size * group_id;
	__global Tuple* begin_input = input + begin;
	__global VAL_TYPE* begin_output = output + begin;
	unsigned int iterations = (partition_size + local_size - 1) / local_size;
	for( int i = 0; i < iterations; i++) {
	
		unsigned int input_id = i * local_size + local_id;

		if(input_id < partition_size){
			//Tuple tup;
			//tup.key = begin_input[input_id].key;
			//tup.val_array[0] = begin_input[input_id].val_array[0];
			Tuple tup = begin_input[input_id];
			begin_output[input_id] = column ? tup.val_array[0] : tup.key;
		}
	}
}

__kernel void Selection(__global Tuple* input, __global Tuple* output, const unsigned int elements, 
						__global VAL_TYPE* histogram, __local Tuple* buffer, const VAL_TYPE threshold)
{  	
   int global_size = get_global_size(0);
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int group_id = get_group_id(0);
	int groups = get_num_groups(0);

	const unsigned int partitions    = groups;
	//add partitions - 1 for rounding purposes
	unsigned int chunkSize = (elements + partitions - 1) / partitions;
	unsigned int partitionSize = (group_id != global_size - 1) ? chunkSize : (elements - group_id  * chunkSize);


	unsigned int begin = chunkSize * group_id;
    __global Tuple* begin_input = input + begin;
    __global Tuple* begin_output = output + begin;

	unsigned int iterations = (partitionSize + local_size - 1) / local_size;
	unsigned int outputIndex = 0;

	for(unsigned int i = 0; i < iterations; ++i)
	{
		VAL_TYPE registers_key;
		VAL_TYPE registers_val;
		
		// coalesced load into registers
		// ??		
		unsigned int input_id = i * local_size	+ local_id;
        unsigned int match = 0;
		

		if(input_id < partitionSize)
		{
			registers_key = begin_input[input_id].key;
			registers_val = begin_input[input_id].val_array[0];
			match = registers_key < threshold ? 1 : 0;
		}
		unsigned int match1 = match;
       barrier(CLK_LOCAL_MEM_FENCE);
		
		// stream compaction in shared memory
		unsigned int values = 0;
		
		unsigned int max = 0;
		__local unsigned int _array[CTAS+1];
	
		if(local_id == 0) _array[0] = values;
		__local unsigned int* array = _array + 1;

		array[local_id] = match;

		barrier(CLK_LOCAL_MEM_FENCE);

		if (CTAS >   1) { if(local_id >=   1) 
			{ unsigned int tmp = array[local_id -   1]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >   2) { if(local_id >=   2) 
			{ unsigned int tmp = array[local_id -   2]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >   4) { if(local_id >=   4) 
			{ unsigned int tmp = array[local_id -   4]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >   8) { if(local_id >=   8) 
			{ unsigned int tmp = array[local_id -   8]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >  16) { if(local_id >=  16) 
			{ unsigned int tmp = array[local_id -  16]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >  32) { if(local_id >=  32) 
			{ unsigned int tmp = array[local_id -  32]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >  64) { if(local_id >=  64) 
			{ unsigned int tmp = array[local_id -  64]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS > 128) { if(local_id >= 128) 
			{ unsigned int tmp = array[local_id - 128]; match = tmp + match; } 
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS > 256) { if(local_id >= 256) 
			{ unsigned int tmp = array[local_id - 256]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }  
		if (CTAS > 512) { if(local_id >= 512) 
			{ unsigned int tmp = array[local_id - 512]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }  

		max = array[CTAS - 1];
		VAL_TYPE output_id = _array[local_id];

		barrier(CLK_LOCAL_MEM_FENCE);
		
		values = max;
		
		if(match1 == 1){
			buffer[output_id].key = registers_key;
			buffer[output_id].val_array[0] = registers_val;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		// write out to global memory
		if (local_id < max)	
		{
			begin_output[outputIndex + local_id].key = buffer[local_id].key;
			begin_output[outputIndex + local_id].val_array[0] = buffer[local_id].val_array[0];
		}

		outputIndex += max;
	}
	 if(local_id == 0) histogram[group_id] = outputIndex;
}

__kernel void Sum(__global unsigned int* histogram, __global unsigned int* totals)
{
	uint globalId = get_global_id(0);
	uint groupId = get_group_id(0);
	uint globalSize = get_global_size(0);
	uint groupSize = get_num_groups(0);

	histogram[globalId] += totals[groupId];
	if(globalId == 0)
		histogram[globalSize] = totals[groupSize];
}

__kernel void PrefixSum(__global unsigned int* input, __global unsigned int* totals,
						const unsigned int isTotal)
{
printf("Beginning PrefixSum\n");
	unsigned int carryIn = 0;
	uint local_id = get_local_id(0);
	uint global_id = get_global_id(0);
	uint global_size = get_global_size(0);
	uint group_id = get_group_id(0);
	uint local_size = get_local_size(0);
	
	__local unsigned int _array[CTAS+1];
	if(local_id == 0) _array[0] = carryIn;
	__local unsigned int* array = _array + 1;
	
	array[local_id] = input[global_id]; 
	unsigned int val = array[local_id];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (CTAS >   1) { if(local_id >=   1) 
		{ unsigned int tmp = array[local_id -   1]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val;
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS >   2) { if(local_id >=   2) 
		{ unsigned int tmp = array[local_id -   2]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val;
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS >   4) { if(local_id >=   4) 
		{ unsigned int tmp = array[local_id -   4]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val; 
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS >   8) { if(local_id >=   8) 
		{ unsigned int tmp = array[local_id -   8]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val; 
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS >  16) { if(local_id >=  16) 
		{ unsigned int tmp = array[local_id -  16]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val; 
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS >  32) { if(local_id >=  32) 
		{ unsigned int tmp = array[local_id -  32]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val; 
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS >  64) { if(local_id >=  64) 
		{ unsigned int tmp = array[local_id -  64]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val; 
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS > 128) { if(local_id >= 128) 
		{ unsigned int tmp = array[local_id - 128]; val = tmp + val; } 
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val; 
			barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTAS > 256) { if(local_id >= 256) 
		{ unsigned int tmp = array[local_id - 256]; val = tmp + val; }
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val;
			barrier(CLK_LOCAL_MEM_FENCE); }  
	if (CTAS > 512) { if(local_id >= 512) 
		{ unsigned int tmp = array[local_id - 512]; val = tmp + val; }
			barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = val; 
			barrier(CLK_LOCAL_MEM_FENCE); }  

	if(local_id == 0){
		if(group_id == get_num_groups(0)-1){
			input[global_size] = _array[local_size];
		}
		if(isTotal == 1){
		
			totals[group_id]	= array[CTAS-1];
		}
	}
	input[global_id] = _array[local_id];

}

__kernel void Gather(
	__global Tuple* begin,__global Tuple* inBegin, const unsigned int size, __global VAL_TYPE* histogram)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int group_id = get_group_id(0);
	int groups = get_num_groups(0);

	const unsigned int partitions    = groups;
	
	unsigned int chunkSize = (size +partitions - 1)/ partitions;

	__global Tuple* inWindowBegin = inBegin + group_id * chunkSize;

	unsigned int beginIndex = histogram[group_id];
	unsigned int endIndex   = histogram[group_id + 1];
	
	unsigned int elements = endIndex - beginIndex;

	unsigned int start = local_id;
	unsigned int step  = local_size;
	
	for(unsigned int i = start; i < elements; i += step)
	{
		begin[beginIndex + i].key = inWindowBegin[i].key;
		begin[beginIndex + i].val_array[0] = inWindowBegin[i].val_array[0];
		
	}
 

}

unsigned int LowerBound(VAL_TYPE key,
	__global Tuple* begin, 
	__global Tuple* end) 
{
	unsigned int low = 0;
	unsigned int high = end - begin;
	
	while(low < high)
	{
		unsigned int median = (low + (high - low) / 2);
		VAL_TYPE masked = begin[median].key;
		if(masked < key)
		{
			low = median + 1;
		}
		else
		{
			high = median;
		}
	}
	
	return low;
}

unsigned int LowerBoundLocal(VAL_TYPE key,
	__local const Tuple* begin,
	__local const Tuple* end)
{
	unsigned int low = 0;
	unsigned int high = end - begin;
	
	while(low < high)
	{
		unsigned int median = (low + (high - low) / 2);
		VAL_TYPE masked = begin[median].key;
		if(masked < key)
		{
			low = median + 1;
		}
		else
		{
			high = median;
		}
	}
	
	return low;
}

unsigned int UpperBound(VAL_TYPE key,
	__global Tuple* begin, 
	__global Tuple* end) 
{
	unsigned int low = 0;
	unsigned int high = end - begin;
	
	while(low < high)
	{
		unsigned int median = (low + (high - low) / 2);
		VAL_TYPE masked = begin[median].key;
		if(key < masked)
		{
			high = median;
		}
		else
		{
			low = median + 1;
		}
	}
	
	return low;
}

unsigned int UpperBoundLocal(VAL_TYPE key,
	__local const Tuple* begin,
	__local const Tuple* end)
{
	unsigned int low = 0;
	unsigned int high = end - begin;
	
	while(low < high)
	{
		unsigned int median = (low + (high - low) / 2);
		VAL_TYPE masked = begin[median].key;
		if(key < masked)
		{
			high = median;
		}
		else
		{
			low = median + 1;
		}
	}
	
	return low;
}

__kernel void FindBounds(
	__global Tuple*      leftBegin, 
	__global Tuple*      rightBegin,
	__global unsigned int* lowerBounds,
	__global unsigned int* upperBounds,
	__global unsigned int* outBounds,
	const unsigned int     leftElements,
	const unsigned int     rightElements
)
{

	int local_id = get_local_id(0);
	int global_id = get_global_id(0);
	int local_size = get_local_size(0);
	int global_size = get_global_size(0);
	int group_id = get_group_id(0);
	int group_size = get_num_groups(0);
	unsigned int start = global_id;
	unsigned int steps  = global_size;

	const unsigned int partitions    = group_size;
	const unsigned int partitionSize = (leftElements + partitions - 1) / partitions;

	for(unsigned int i = start; i < partitions; i += steps)
	{

		unsigned int leftIndex = partitionSize * i;
		unsigned int leftMax   = MIN(partitionSize * (i + 1), leftElements);
		if(leftIndex < leftElements)
		{
			VAL_TYPE key = leftBegin[leftIndex].key;
			unsigned int lBound = LowerBound(key, rightBegin, rightBegin + rightElements);
			lowerBounds[i] = lBound;

			unsigned int uBound = 0;
			if(leftMax < leftElements) 
			{
				VAL_TYPE key = leftBegin[leftMax-1].key;
				uBound = UpperBound(key, rightBegin, rightBegin + rightElements);
			}
			else
			{
				uBound = rightElements;
			}

		        upperBounds[i] = uBound;
				outBounds[i]   = 4 * MAX((leftMax - leftIndex), (uBound - lBound)); // guessing the output size of each partition
		}
		else
		{
			lowerBounds[i] = rightElements;
			upperBounds[i] = rightElements;
			outBounds[i]   = 0;
		}
	}
}

void Dma(__local Tuple* out, __global Tuple* in, __global Tuple* inEnd)
{

	unsigned int elements = inEnd - in;
	
	for(unsigned int i = get_local_id(0); i < elements; i+=get_local_size(0))
	{
		out[i].key = in[i].key;
		out[i].val_array[0] = in[i].val_array[0];
		out[i].val_array[1] = in[i].val_array[1];
		out[i].val_array[2] = in[i].val_array[2];
	}
	
}

__global Tuple* Dma_new(__global Tuple* out, __local Tuple* in, __local Tuple* inEnd)
{

	unsigned int elements = inEnd - in;
	
	for(unsigned int i = get_local_id(0); i < elements; i+=get_local_size(0))
	{
		out[i].key = in[i].key;
		out[i].val_array[0] = in[i].val_array[0];
		out[i].val_array[1] = in[i].val_array[1];
		out[i].val_array[2] = in[i].val_array[2];
	}	
	return out + elements;

}

__kernel void Product(__global Tuple* left_input, __global Tuple* right_input, __global prod_tuple* output,
			__local Tuple* right_local, const unsigned int left_elements, const unsigned int right_elements)
{
	int global_id = get_global_id(0);
	int global_size = get_global_size(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int group_id = get_group_id(0);
	int groups = get_num_groups(0);

	const unsigned int partitions = groups;
	unsigned int chunk_size = (left_elements + partitions - 1) / partitions;
	unsigned int partition_size = (group_id != global_size - 1) ? chunk_size : (left_elements - group_id * chunk_size);

	for (unsigned int i = 0; i < right_elements; i++) {
		 right_local[i] = right_input[i];
	}

	unsigned int beginI = chunk_size * group_id;
	__global Tuple* begin_left = left_input + beginI;
	unsigned int beginO = chunk_size * right_elements * group_id;
	__global prod_tuple* begin_output = output + beginO; 

	unsigned int iterations = (partition_size + local_size - 1) / local_size;
	unsigned int output_index = 0;

	for (unsigned int i = 0; i < iterations; ++i)
	{
		for (unsigned int j = 0; j < right_elements; j++)
		{
			VAL_TYPE left_key, left_val;
			VAL_TYPE right_key, right_val;

			//Is Coalesced Load the best method?
			//Probably best if input does not fit in shared memory i.e. Select
			//Might not make a difference if input fits in shared memory i.e. Product
			unsigned int input_id = i * local_size + local_id;
			
			if (input_id < partition_size)
			{
				left_key = begin_left[input_id].key;
				left_val = begin_left[input_id].val_array[0];
				right_key = right_local[j].key;
				right_val = right_local[j].val_array[0];
			}
			//printf("\n%u %u %u %u %u %u %u %u %u\n", group_id, local_id, input_id, i, j, left_key, left_val, right_key, right_val);
/*
			__local unsigned int _array[CTAS+1];

			if (local_id == 0) _array[0] = 0;
			__local unsigned int* array = _array + 1;

			array[local_id] = match;
*/
			barrier(CLK_LOCAL_MEM_FENCE);

			unsigned int output_id = input_id * right_elements + j;

			begin_output[output_id].key_array[0] = left_key;
			begin_output[output_id].val_array[0] = left_val;
			begin_output[output_id].key_array[1] = right_key;
			begin_output[output_id].val_array[1] = right_val;
		}
	}
}

__kernel void Unique(
	__global VAL_TYPE* input,
	__global VAL_TYPE* output,
	__local  VAL_TYPE* buffer,
	__global unsigned int* histogram,
	const unsigned int elements)
{
printf("Beginning Unique\n");
	int global_id   = get_global_id(0);
	int global_size = get_global_size(0);
	int local_id    = get_local_id(0);
	int local_size  = get_local_size(0);
	int group_id    = get_group_id(0);
	int groups      = get_num_groups(0);

	const unsigned int partitions = groups;
	unsigned int chunk_size = (elements + partitions - 1) / partitions;
	unsigned partition_size = (group_id != global_size - 1) ? chunk_size : (elements - group_id * chunk_size);

	//unsigned int begin = chunk_size * group_id;
	unsigned int begin = chunk_size * group_id;
	__global VAL_TYPE* begin_input = input + begin;
	__global VAL_TYPE* begin_output = output + begin;

	unsigned int iterations = (partitions + local_size - 1) / local_size;
	unsigned int output_index = 0;

	for (unsigned int i = 0; i < iterations; i++) {
		VAL_TYPE input1, input2;
		unsigned int input_id = i * local_size + local_id;
		unsigned int match = 1;

		input1 = begin_input[input_id];
		if (input_id < partition_size - 1) {
			input1 = begin_input[input_id];
			input2 = begin_input[input_id + 1];
			match = input1 == input2 ? 1 : 0;
		}
		unsigned int mathc1 = match;
		barrier(CLK_LOCAL_MEM_FENCE);

		unsigned int values = 0;
		unsigned int max = 0;

		__local unsigned int _array[CTAS+1];

		if (local_id == 0) _array[0] = values;
		__local unsigned int* array = _array + 1;

		array[local_id] = match;

		barrier(CLK_LOCAL_MEM_FENCE);

		if (CTAS >   1) { if(local_id >=   1)
			{ unsigned int tmp = array[local_id -   1]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >   2) { if(local_id >=   2)
			{ unsigned int tmp = array[local_id -   2]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >   4) { if(local_id >=   4)
			{ unsigned int tmp = array[local_id -   4]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >   8) { if(local_id >=   8)
			{ unsigned int tmp = array[local_id -   8]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >  16) { if(local_id >=  16)
			{ unsigned int tmp = array[local_id -  16]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >  32) { if(local_id >=  32)
			{ unsigned int tmp = array[local_id -  32]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS >  64) { if(local_id >=  64)
			{ unsigned int tmp = array[local_id -  64]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS > 128) { if(local_id >= 128)
			{ unsigned int tmp = array[local_id - 128]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS > 256) { if(local_id >= 256)
			{ unsigned int tmp = array[local_id - 256]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }
		if (CTAS > 512) { if(local_id >= 512)
			{ unsigned int tmp = array[local_id - 512]; match = tmp + match; }
				barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = match; barrier(CLK_LOCAL_MEM_FENCE); }

		max = array[CTAS - 1];
		VAL_TYPE output_id = _array[local_id];
		barrier(CLK_LOCAL_MEM_FENCE);

		values = max;

		if (match == 1) {
			buffer[output_id] = input1;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		//if (match == 1) buffer[output_id] = input1;

		// ??? What is going on in these next few lines
		if (local_id < max) {
			begin_output[output_index + local_id] = buffer[local_id];
		}
		output_index += max;
	}
	if (local_id == 0) histogram[group_id] = output_index;
}

__kernel void CornerCase(
	__global VAL_TYPE* output,
	__global unsigned int* histogram,
	const unsigned int localSize)
{
printf("Beginning CornerCase\n");
	unsigned int gid = get_global_id(0);
	
	unsigned int index1 = gid*localSize + (histogram[gid] - 1);
	unsigned int index2 = (gid+1) * localSize;

	VAL_TYPE temp1 = output[index1];
	VAL_TYPE temp2 = output[index2];

	if (temp1 == temp2) {
		unsigned int val = histogram[gid] - 1;
		histogram[gid] = val;
	}
}

__kernel void UniqueGather(
	__global VAL_TYPE* begin,
	__global VAL_TYPE* in_begin,
	__global VAL_TYPE* histogram)
{
printf("Beginning UniqueGather\n");
	unsigned int group_id = get_group_id(0);

	__global VAL_TYPE* in_window_begin = in_begin + (get_local_size(0) * group_id);

	unsigned int begin_index = histogram[group_id];
	unsigned int end_index   = histogram[group_id + 1];
	unsigned int elements    = end_index - begin_index;

	unsigned int start = get_local_id(0);
	unsigned int step = get_local_size(0);
printf("Beginning Scan\n");
	for (unsigned int i = start; i < elements; i += step) {
		begin[begin_index + i] = in_window_begin[i];
	}
}

__kernel void Join(
	__global Tuple*      leftBegin,
	__global Tuple*      rightBegin,
	__global unsigned int* lowerBounds,
	__global unsigned int* upperBounds,
	__global unsigned int* outBounds,
	const unsigned int     leftElements,
	const unsigned int     rightElements,	
	__global Tuple*   output,
	__global unsigned int* histogram,
	const unsigned int first )
{

	int local_size = get_local_size(0);
	__local Tuple leftCache[THREADS]; 
	__local Tuple rightCache[THREADS]; 

	barrier(CLK_LOCAL_MEM_FENCE);
	
	unsigned int id = get_group_id(0);
	const unsigned int partitions    = get_num_groups(0);
	const unsigned int partitionSize = (leftElements + partitions - 1) / partitions;
	__global Tuple* l    = leftBegin + MIN(partitionSize * id,       leftElements);
	__global Tuple* lend = leftBegin + MIN(partitionSize * (id + 1), leftElements);

	__global Tuple* r    = rightBegin + lowerBounds[id];
	__global Tuple* rend = rightBegin + upperBounds[id];

	__global Tuple* oBegin = output + outBounds[id] - outBounds[0];
	__global Tuple* o      = oBegin;

	while(l != lend && r != rend)
	{

		unsigned int rightBlockSize = MIN(rend - r, THREADS); //* unroll);
		unsigned int leftBlockSize  = MIN(lend - l, THREADS); // * unroll);

		__global Tuple* leftBlockEnd  = l + leftBlockSize;
		__global Tuple* rightBlockEnd = r + rightBlockSize;
		
		Dma(leftCache,  l, leftBlockEnd );
		Dma(rightCache, r, rightBlockEnd);

		barrier(CLK_LOCAL_MEM_FENCE);

		Tuple lMax = *(leftCache + leftBlockSize - 1);
		Tuple rMin = *rightCache;


		VAL_TYPE lMaxValue = lMax.key;
		VAL_TYPE rMinValue = rMin.key;

		if(lMaxValue < rMinValue)
		{
			l = leftBlockEnd;
		}
		else
		{
			Tuple lMin = *leftCache;
			Tuple rMax = *(rightCache + rightBlockSize - 1);
			
			VAL_TYPE lMinValue = lMin.key;
			VAL_TYPE rMaxValue = rMax.key;

			if(rMaxValue < lMinValue)
			{
				r = rightBlockEnd;
			}
			else
			{
				__local Tuple cache[CACHE_SIZE];
				__local unsigned int _array[CTAS+1];
				
				int local_id = get_local_id(0);

				__local const Tuple* r = rightCache + local_id;

				Tuple rTuple;
				VAL_TYPE lKey = 0;
				VAL_TYPE lValue = 0;
				// JY replace - const unsigned int rightElements = rightCache + rightBlockSize - rightCache;
				const unsigned int rightElements = rightBlockSize;
				
				unsigned int foundCount = 0;	
				unsigned int foundCount1 = 0;

				unsigned int lower = 0;
				unsigned int higher = 0;

				if(local_id < rightElements)
				{
					rTuple = *r;
					VAL_TYPE rKey = rTuple.key;
					lower  = LowerBoundLocal(rKey, leftCache, leftCache  + leftBlockSize);
					higher = UpperBoundLocal(rKey, leftCache, leftCache  + leftBlockSize);
					
					foundCount = higher - lower;
					foundCount1 = foundCount;
				}
				
				barrier(CLK_LOCAL_MEM_FENCE);

				unsigned int total = 0;
				unsigned int index = 1;
				

				unsigned int carryIn = 0;
	
				if(local_id == 0) _array[0] = carryIn;
				__local unsigned int* array = _array + 1;

				array[local_id] = foundCount;

				barrier(CLK_LOCAL_MEM_FENCE);

				if (CTAS >   1) { if(local_id >=   1) 
					{ unsigned int tmp = array[local_id -   1]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS >   2) { if(local_id >=   2) 
					{ unsigned int tmp = array[local_id -   2]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS >   4) { if(local_id >=   4) 
					{ unsigned int tmp = array[local_id -   4]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS >   8) { if(local_id >=   8) 
					{ unsigned int tmp = array[local_id -   8]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS >  16) { if(local_id >=  16) 
					{ unsigned int tmp = array[local_id -  16]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS >  32) { if(local_id >=  32) 
					{ unsigned int tmp = array[local_id -  32]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS >  64) { if(local_id >=  64) 
					{ unsigned int tmp = array[local_id -  64]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS > 128) { if(local_id >= 128) 
					{ unsigned int tmp = array[local_id - 128]; foundCount = tmp + foundCount; } 
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
				if (CTAS > 256) { if(local_id >= 256) 
					{ unsigned int tmp = array[local_id - 256]; foundCount = tmp + foundCount; }
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }  
				if (CTAS > 512) { if(local_id >= 512) 
					{ unsigned int tmp = array[local_id - 512]; foundCount = tmp + foundCount; }
						barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }  

				total = array[CTAS - 1];
				index = _array[local_id];

				barrier(CLK_LOCAL_MEM_FENCE);

				if(total <= CACHE_SIZE)
				{
		
					for(unsigned int c = 0; c < foundCount1; ++c)
					{
						lKey = leftCache[lower+c].key;
						lValue = leftCache[lower+c].val_array[0];
						VAL_TYPE lValue1 = leftCache[lower+c].val_array[1];
						if(first == 1){
							cache[index + c].key = lKey;
							cache[index + c].val_array[0] = lValue;
							cache[index + c].val_array[1] = rTuple.val_array[0];
						}
						else{
							cache[index + c].key = lKey;
							cache[index + c].val_array[0] = lValue;
							cache[index + c].val_array[1] = lValue1;
							cache[index + c].val_array[2] = rTuple.val_array[0];
						}
					}
		
					barrier(CLK_LOCAL_MEM_FENCE);
		
					Dma_new(o, cache, cache + total);
				}

				else
				{
					//cache not big enough
					__local unsigned int sharedCopiedThisTime;
	
					unsigned int copiedSoFar = 0;
					bool done = false;
					
	
					while(copiedSoFar < total)
					{
						if(index + foundCount1 <= CACHE_SIZE && !done)
						{
							for(unsigned int c = 0; c < foundCount1; ++c)
							{
								lKey = leftCache[lower+c].key;
								lValue = leftCache[lower+c].val_array[0];
								VAL_TYPE lValue1 = leftCache[lower+c].val_array[1];
								if(first == 1){
									cache[index + c].key = lKey;
									cache[index + c].val_array[0] = lValue;
									cache[index + c].val_array[1] = rTuple.val_array[0];
								}
								else{
									cache[index + c].key = lKey;
									cache[index + c].val_array[0] = lValue;
									cache[index + c].val_array[1] = lValue1;
									cache[index + c].val_array[2] = rTuple.val_array[0];
								}
							}
					
							done = true;
						}
			
						if(index <= CACHE_SIZE && index + foundCount1 > CACHE_SIZE) //overbounded thread
          				{
							sharedCopiedThisTime = index;
				
           				}
						else if (get_local_id(0) == THREADS - 1 && done){
							sharedCopiedThisTime = index + foundCount1;
						}
		
						barrier(CLK_LOCAL_MEM_FENCE);
		
						unsigned int copiedThisTime = sharedCopiedThisTime;

						index -= copiedThisTime;
						copiedSoFar += copiedThisTime;
		
						o = Dma_new(o, cache, cache + copiedThisTime);
            
					}

				}


				unsigned int joined = total;


				o += joined;
				__global Tuple* ri = rightBlockEnd;

				for(; ri != rend;)
				{
					rightBlockSize = MIN(THREADS, rend - rightBlockEnd);
					rightBlockEnd  = rightBlockEnd + rightBlockSize;
					Dma(rightCache, ri, rightBlockEnd);
				
					barrier(CLK_LOCAL_MEM_FENCE);
			
					rMin = *rightCache;
   					rMinValue = rMin.key;

					if(lMaxValue < rMinValue) break;
	                
					int local_id = get_local_id(0);
					__local const Tuple* r = rightCache + local_id;


					Tuple rTuple;
					VAL_TYPE lKey = 0;
					VAL_TYPE lValue = 0;
					const unsigned int rightElements = rightCache + rightBlockSize - rightCache;
					
					unsigned int foundCount = 0;	
					unsigned int foundCount1 = 0;

					unsigned int lower = 0;
					unsigned int higher = 0;

					if(local_id < rightElements)
					{
						rTuple = *r;
						VAL_TYPE rKey = rTuple.key;
						lower  = LowerBoundLocal(rKey, leftCache, leftCache  + leftBlockSize);
						higher = UpperBoundLocal(rKey, leftCache, leftCache  + leftBlockSize);
						foundCount = higher - lower;
						foundCount1 = foundCount;
					}
	
					barrier(CLK_LOCAL_MEM_FENCE);

					unsigned int total = 0;
					unsigned int index = 1;
					

					unsigned int carryIn = 0;
	
					if(local_id == 0) _array[0] = carryIn;
					__local unsigned int* array = _array + 1;

					array[local_id] = foundCount;

					barrier(CLK_LOCAL_MEM_FENCE);

					if (CTAS >   1) { if(local_id >=   1) 
						{ unsigned int tmp = array[local_id -   1]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS >   2) { if(local_id >=   2) 
						{ unsigned int tmp = array[local_id -   2]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS >   4) { if(local_id >=   4) 
						{ unsigned int tmp = array[local_id -   4]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS >   8) { if(local_id >=   8) 
						{ unsigned int tmp = array[local_id -   8]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS >  16) { if(local_id >=  16) 
						{ unsigned int tmp = array[local_id -  16]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS >  32) { if(local_id >=  32) 
						{ unsigned int tmp = array[local_id -  32]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS >  64) { if(local_id >=  64) 
						{ unsigned int tmp = array[local_id -  64]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS > 128) { if(local_id >= 128) 
						{ unsigned int tmp = array[local_id - 128]; foundCount = tmp + foundCount; } 
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }
					if (CTAS > 256) { if(local_id >= 256) 
						{ unsigned int tmp = array[local_id - 256]; foundCount = tmp + foundCount; }
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }  
					if (CTAS > 512) { if(local_id >= 512) 
						{ unsigned int tmp = array[local_id - 512]; foundCount = tmp + foundCount; }
							barrier(CLK_LOCAL_MEM_FENCE); array[local_id] = foundCount; barrier(CLK_LOCAL_MEM_FENCE); }  

					total = array[CTAS - 1];
					index = _array[local_id];

					barrier(CLK_LOCAL_MEM_FENCE);

					if(total <= CACHE_SIZE)
					{
		
						for(unsigned int c = 0; c < foundCount1; ++c)
						{
							lKey = leftCache[lower+c].key;
							lValue = leftCache[lower+c].val_array[0];
							VAL_TYPE lValue1 = leftCache[lower+c].val_array[1];
							if(first == 1){
								cache[index + c].key = lKey;
								cache[index + c].val_array[0] = lValue;
								cache[index + c].val_array[1] = rTuple.val_array[0];
							}
							else{
								cache[index + c].key = lKey;
								cache[index + c].val_array[0] = lValue;
								cache[index + c].val_array[1] = lValue1;
								cache[index + c].val_array[2] = rTuple.val_array[0];
							}
						}
		
						barrier(CLK_LOCAL_MEM_FENCE);
		
						Dma_new(o, cache, cache + total);
					}

					else
					{
						//cache not big enough
						__local unsigned int sharedCopiedThisTime;
	
						unsigned int copiedSoFar = 0;
						bool done = false;
						
	
						while(copiedSoFar < total)
						{
							if(index + foundCount1 <= CACHE_SIZE && !done)
							{
								for(unsigned int c = 0; c < foundCount1; ++c)
								{
									lKey = leftCache[lower+c].key;
									lValue = leftCache[lower+c].val_array[0];
									VAL_TYPE lValue1 = leftCache[lower+c].val_array[1];
									if(first == 1){
										cache[index + c].key = lKey;
										cache[index + c].val_array[0] = lValue;
										cache[index + c].val_array[1] = rTuple.val_array[0];
									}
									else{
										cache[index + c].key = lKey;
										cache[index + c].val_array[0] = lValue;
										cache[index + c].val_array[1] = lValue1;
										cache[index + c].val_array[2] = rTuple.val_array[0];
									}
								}
			
								done = true;
							}
			
							if(index <= CACHE_SIZE && index + foundCount1 > CACHE_SIZE) //overbounded thread
          					{
								sharedCopiedThisTime = index;
				
           					}
							else if (get_local_id(0) == THREADS - 1 && done){
								sharedCopiedThisTime = index + foundCount1;
							}
		
							barrier(CLK_LOCAL_MEM_FENCE);
		
							unsigned int copiedThisTime = sharedCopiedThisTime;

							index -= copiedThisTime;
							copiedSoFar += copiedThisTime;
		
							o = Dma_new(o, cache, cache + copiedThisTime);
            
						}

					}


					joined = total;


					o += joined;
					ri = rightBlockEnd;
				}
				
				l = leftBlockEnd;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		 }
	}

	if(get_local_id(0) == 0) 
	{
		histogram[id] = o - oBegin;
	}
	
}

__kernel void GatherJoin(
	__global Tuple*   begin,
	__global Tuple*   inBegin,
	__global unsigned int* outBounds,
	__global unsigned int* histogram)
{
	int group_id = get_group_id(0);	
	
	__global Tuple* inWindowBegin = inBegin + outBounds[group_id];
	
	unsigned int beginIndex = histogram[group_id];
	unsigned int endIndex   = histogram[group_id + 1];
	unsigned int elements = endIndex - beginIndex;

	unsigned int start = get_local_id(0);
	unsigned int steps  = get_local_size(0);

	for(unsigned int i = start; i < elements; i += steps)
	{
		begin[beginIndex + i].key = inWindowBegin[i].key;
		begin[beginIndex + i].val_array[0] = inWindowBegin[i].val_array[0];
		begin[beginIndex + i].val_array[1] = inWindowBegin[i].val_array[1];
		begin[beginIndex + i].val_array[2] = inWindowBegin[i].val_array[2];
	}
 

}
