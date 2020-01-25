
/********************************************************************
*	By: Khaled Alshatti												*
*	Project 3														*
*	Date: 07/26/2019												*
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

void checkErrorCuda(cudaError_t errors, const char out[])
{
    if (errors != cudaSuccess)
    {
        printf("Cuda Error occured: %s, %s, \n", out, cudaGetErrorString(errors));
        exit(EXIT_FAILURE);
    }
}

//data generator
void dataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

//define the histogram kernel here
//global memory coalesced memory access with interleaving partitioning, so striding
__global__ void histogram(int * inData, int rSize, int numPartition, int *outData )
{
    int k, h;
    int block, stride;

    block = blockIdx.x * blockDim.x + threadIdx.x;
    stride = blockDim.x * gridDim.x; 

    for (k = block; k < rSize; k+=stride)
    {
        h = bfe(inData[k], 0, numPartition); 
        //outData[h]++;
		atomicAdd(&outData[h], 1); 
    }
}

//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
__global__ void prefixScan(int* inData, int width, int* outPrefix)
{
    extern __shared__ int sum[];
    int thread, i, ai, bi;

    i = 1;
    thread = threadIdx.x;

    if(thread < width)
    {
        if(thread == 0)
        {
            sum[thread] = 0;
        }
        else
        {
            sum[thread] = inData[thread-1];
        }
        __syncthreads();

        while(i< width)
        {
            ai = thread - i;
            bi = thread;
            
            if(ai > 0)
            {
                sum[bi] += sum[ai];
            }
            else
            {
                sum[bi] = sum[bi];
            }

            __syncthreads();
            i *= 2;
        }
        outPrefix[thread] = sum[thread];
    }
}

//define the reorder kernel here
__global__ void Reorder(int *inData, int rSize, int numPartition, int *outPrefix, int *outReorder)
{
    int k, h;
    int offset;    
    int block, stride;

    block = blockIdx.x * blockDim.x + threadIdx.x;
    stride = blockDim.x * gridDim.x; 

    for (k = block; k < rSize; k+=stride)
    {
        k = inData[k];
        h = bfe(k, 0, numPartition); 
        offset = atomicAdd((int*)outPrefix[h], 1);
		atomicAdd(&outReorder[offset], 1); 
    }
}

int main(int argc, char const *argv[])
{
    int i, rSize, numPartitions, count, bits;
    int *r_h, *histo, *prefix, *reorder; 
    int blocksize = 64;

    if (argc != 3 )
	{
		printf("Missing Input, Proper Format:\n./proj3-khalshatti {#of_elements_in_array} {#of_partitions}\n\n");
		exit(0);
    }
    
    rSize = atoi(argv[1]);
    numPartitions = atoi(argv[2]);

    count = (int)ceil( (double)rSize/ (double) blocksize);
    bits =(int)log2((double)numPartitions);

    checkErrorCuda(cudaMallocHost((void**)&r_h, sizeof(int)*rSize), "Use Pinned Memory" ); //use pinned memory 
    
    //data generation
    dataGenerator(r_h, rSize, 0, 1);

    //**************************************************//
    //your code                                         //  
    //**************************************************//

    //check for cuda error when allocating histogram, prefix, reorder
    checkErrorCuda(cudaMallocHost((void **) &histo, sizeof(int)* numPartitions), "Cuda Malloc Histogram");
    checkErrorCuda(cudaMallocHost((void **) &prefix, sizeof(int)* numPartitions), "Cuda Malloc Prefix");
    checkErrorCuda(cudaMallocHost((void **) &reorder, sizeof(int)* rSize), "Cuda Malloc Reorder");
 
    //Measuring Running time
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );
    
    //call histogram kernel
    histogram<<<count, blocksize>>>(r_h, rSize, bits, histo);

    checkErrorCuda(cudaDeviceSynchronize(), "histogram" );

    //call Prefix Scan kernel
    prefixScan<<<1, numPartitions, sizeof(int)*numPartitions>>>(histo, numPartitions, prefix);

    checkErrorCuda(cudaDeviceSynchronize(), "prefixScan" );

    //call Reorder kernel
    Reorder<<<count, blocksize>>>(r_h, rSize, numPartitions, prefix, reorder);

    //output
    for(i = 0; i < numPartitions; i++)
    {
        printf("Partition %d:   Offset: %d  Number of Keys: %d\n", i, prefix[i], histo[i]);
    }
    printf("\n");

    //Measuring Running time
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop ); 
    printf( "Time to generate: %0.5f ms\n", elapsedTime );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    //freeing r_h, histo, prefix, reorder
    cudaFreeHost(r_h);
    cudaFreeHost(histo);
    cudaFreeHost(prefix);
    cudaFreeHost(reorder);
    
    return 0;
}
