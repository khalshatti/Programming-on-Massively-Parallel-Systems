/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/

/********************************************************************
*	By: Khaled Alshatti												*
*	Project 2														*
*	Date: 06/30/2019													*
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;

// To check for Cuda Error
void checkErrorCuda(cudaError_t errors, const char out[])
{
    if (errors != cudaSuccess)
    {
        printf("Cuda Error occured: %s, %s, \n", out, cudaGetErrorString(errors));
        exit(EXIT_FAILURE);
    }
}

/* 
	distance of two points in the atom_list 
*/
__device__ double p2p_distance(atom * atomList, int ind1, int ind2) {
	
	double x1 = atomList[ind1].x_pos;
	double x2 = atomList[ind2].x_pos;
	double y1 = atomList[ind1].y_pos;
	double y2 = atomList[ind2].y_pos;
	double z1 = atomList[ind1].z_pos;
	double z2 = atomList[ind2].z_pos;
		
	double x = pow((x1 - x2), 2);
	double y = pow((y1 - y2), 2);
	double z = pow((z1 - z2), 2);

	//return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
	return sqrt(x + y + z);
}

/* 
	brute-force SDH solution in a single GPU thread 
*/
__global__ void PDH_baseline(bucket *histogram, atom * atomList, long long width, int size) {
	int i, j, x, h_pos;
    double dist;
    
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = i + 1;
	
    for(x = j; x < size; x++) 
    {
		dist = p2p_distance(atomList, i, x);
		h_pos = (int) (dist / width);
		//histogram[h_pos].d_cnt++;
		atomicAdd(&histogram[h_pos].d_cnt, 1);
	}
	//wait for everybody to be done
	//__syncthreads();
}

/* 
	set a checkpoint and show the (natural) running time in seconds for CPU
*/
double report_running_time_CPU() {
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff= endTime.tv_usec-startTime.tv_usec;
    if(usec_diff < 0) {
        sec_diff --;
        usec_diff += 1000000;
    }
    printf("Running time for CPU version: %ld.%06lds\n", sec_diff, usec_diff);
    return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/* 
	set a checkpoint and show the (natural) running time in seconds for GPU
*/
double report_running_time_GPU() {
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff= endTime.tv_usec-startTime.tv_usec;
    if(usec_diff < 0) {
        sec_diff --;
        usec_diff += 1000000;
    }
    printf("Running time for GPU version: %ld.%06lds\n", sec_diff, usec_diff);
    return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

__global__ void PDH2D_basline_algo2(bucket *deviceHistogram, atom *deviceAtomList, long long width, double size, int numBlocks) 
{

	
	int i, j, h_pos, block, thread, blockdim, blockid;
	double dist;
	register atom left;
	extern __shared__ atom sharedAtomList[];

	//how many threads in a block (width of block)
	blockid = blockIdx.x;	
	blockdim = blockDim.x;
	thread = threadIdx.x;
	block = blockIdx.x * blockDim.x + threadIdx.x;
	
	//set the b-th input data block loaded to cache
	left = deviceAtomList[block];

	if(block < width)
	for(i = blockid + 1; i < numBlocks; i++)
	{
		//set the i-th input data block loaded to cache
		sharedAtomList[thread] = deviceAtomList[i*blockdim + thread];
		__syncthreads();

		for(j = 0; j < blockdim; j++) 
		{
			if(i*blockdim + j < width) 
			{
				dist = sqrt((left.x_pos - sharedAtomList[j].x_pos)*(left.x_pos - sharedAtomList[j].x_pos) + (left.y_pos - sharedAtomList[j].y_pos)*(left.y_pos - sharedAtomList[j].y_pos) + (left.z_pos - sharedAtomList[j].z_pos)*(left.z_pos - sharedAtomList[j].z_pos));
				h_pos = (int)(dist/size);
				atomicAdd(&deviceHistogram[h_pos].d_cnt,1);
			}
		}
	}

	sharedAtomList[thread] = left;
	
	if(block < width)
	for(i = thread + 1; i < blockdim; i++) 
	{
		if(blockdim * blockid + i < width) 
		{
			dist = sqrt((left.x_pos - sharedAtomList[i].x_pos)*(left.x_pos - sharedAtomList[i].x_pos) + (left.y_pos - sharedAtomList[i].y_pos)*(left.y_pos - sharedAtomList[i].y_pos) + (left.z_pos - sharedAtomList[i].z_pos)*(left.z_pos - sharedAtomList[i].z_pos));
			h_pos = (int)(dist/size);
			atomicAdd(&deviceHistogram[h_pos].d_cnt,1);
		}
	}
}

//SDH with output Privatization
__global__ void PDH2D_basline_algo3(bucket *histogram, atom *atomList, long long width, double size, int numBlocks, int numBuckets) {
	
	int i, j, h_pos, block, thread, blockdim, blockid;
	double dist;
	atom left, right;
	
	extern __shared__ bucket sharedHistogram[];
	__shared__ atom sharedAtomList[256];

	//how many threads in a block (width of block)
	blockid = blockIdx.x;	
	blockdim = blockDim.x;
	thread = threadIdx.x;
	block = blockIdx.x * blockDim.x + threadIdx.x;
	
	//initialize shared memory to zero
	for(i = thread; i < numBuckets; i += blockdim) 
	{
		sharedHistogram[i].d_cnt = 0;
	}
	
	left = atomList[block];
	__syncthreads();
	
	if(block < width) 
	{
		for(i = blockid + 1; i < numBlocks; i++) 
		{		
			//
			sharedAtomList[thread] = atomList[i*blockdim + thread];
			__syncthreads();

			if(i*blockdim < width)
			for(j = 0; j < blockdim; j++) 
			{
				if(i*blockdim + j < width) 
				{
					right = sharedAtomList[j];
					dist = sqrt((left.x_pos - right.x_pos)*(left.x_pos - right.x_pos) + (left.y_pos - right.y_pos)*(left.y_pos - right.y_pos) + (left.z_pos - right.z_pos)*(left.z_pos - right.z_pos));
					h_pos = (int)(dist/size);
					atomicAdd(&sharedHistogram[h_pos].d_cnt,1);
				}
			}
			__syncthreads();
		}
	}

	sharedAtomList[thread] = left;
	__syncthreads();

	if(block < width)
	for(i = thread + 1; i < blockdim; i++) 
	{
		
		if(blockdim * blockid + i < width) 
		{
			right = sharedAtomList[i];
			dist = sqrt((left.x_pos - right.x_pos)*(left.x_pos - right.x_pos) + (left.y_pos - right.y_pos)*(left.y_pos - right.y_pos) + (left.z_pos - right.z_pos)*(left.z_pos - right.z_pos));
			h_pos = (int)(dist/size);
			atomicAdd(&sharedHistogram[h_pos].d_cnt,1);
		}
	}
	__syncthreads();

	for(i = thread; i < numBuckets; i += blockdim) 
	{
		atomicAdd(&histogram[i].d_cnt,sharedHistogram[i].d_cnt);
	}
}

/* 
	SDH solution in GPU thread 

__global__ void PDH4D_baseline(bucket *histogram, atom * atomList, double width) 
{
	int i, j, h_pos;
    double dist;
    
    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(i<j) 
    {
		dist = p2p_distance(atomList, i, j);
		h_pos = (int) (dist / width);
        histogram[h_pos].d_cnt++;
		//atomicAdd(&histogram[h_pos].d_cnt, 1);
		printf("%d, %d: %d, %f \n", i, j, h_pos, dist);
	}
	__syncthreads();
}
*/

/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket * historgram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;
	//int PDH_thr;

	if (argc != 4 )
	{
		printf("Missing Input, Proper Format:\n./proj2-khalshatti {#of_samples} {bucket_width} {block_size}\n\n");
		exit(0);
	}
	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	//PDH_thr = atoi(argv[3]);
	
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	//dim3 thread(PDH_thr);
	dim3 thread(32);
	dim3 grid(ceil((float)PDH_acnt/thread.x));

    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    size_t histogramSize = sizeof(bucket) * num_buckets;
    size_t atomSize = sizeof(atom)*PDH_acnt;

	histogram = (bucket *)malloc(histogramSize);
	atom_list = (atom *)malloc(atomSize);

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	bucket *deviceHistogram = NULL;
	//bucket *deviceHistogram = (bucket *)malloc(histogramSize);
	//atom *deviceAtomList = (atom *)malloc(atomSize);
	atom *deviceAtomList = NULL;
	

	//Check for cuda error
	checkErrorCuda(cudaMalloc((void**) &deviceHistogram, histogramSize), "Cuda Malloc Historgram");
	checkErrorCuda(cudaMalloc((void**) &deviceAtomList, atomSize), "Cuda Malloc Atom List");
	checkErrorCuda(cudaMemcpy(deviceHistogram, histogram, histogramSize, cudaMemcpyHostToDevice), "Copy Histogrm to Device");
	checkErrorCuda(cudaMemcpy(deviceAtomList, atom_list, atomSize, cudaMemcpyHostToDevice), "Copy Atom List to Device");

	/* start counting time */
	gettimeofday(&startTime, &Idunno);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/* calling cuda kernel */
	//PDH_baseline <<<ceil(PDH_acnt/16), 32>>> (deviceHistogram, deviceAtomList, PDH_res, PDH_acnt);
    //PDH4D_baseline <<<ceil(PDH_acnt/PDH_thr), PDH_thr, sizeof(bucket)*PDH_thr>>> (deviceHistogram, deviceAtomList, PDH_acnt, PDH_res, num_buckets);
	PDH2D_basline_algo2<<<grid,thread,num_buckets*sizeof(bucket)>>>(deviceHistogram, deviceAtomList, PDH_acnt, PDH_res, grid.x);
	//PDH2D_basline_algo3<<<grid,thread,num_buckets*sizeof(bucket)>>>(deviceHistogram, deviceAtomList, PDH_acnt, PDH_res, grid.x, num_buckets);

	checkErrorCuda(cudaGetLastError(), "Check last error");
	//checkErrorCuda(cudaMemcpy(histogram, deviceHistogram, histogramSize, cudaMemcpyDeviceToHost), "Copy Device Histogram");

	/* check the total running time */ 
	//report_running_time();
	report_running_time_CPU();
	report_running_time_GPU();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("**********Total Running Time of Kernel: %0.5f ms*********** \n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkErrorCuda(cudaMemcpy(histogram, deviceHistogram, histogramSize, cudaMemcpyDeviceToHost), "Copy Device Histogram");

	/* print out the histogram */
	output_histogram(histogram);

	checkErrorCuda(cudaFree(deviceHistogram), "Free Device Historgram");
	checkErrorCuda(cudaFree(deviceAtomList),"Free Device Atom List");

	//Free Memory
	cudaFree(histogram);
	cudaFree(atom_list);

	checkErrorCuda(cudaDeviceReset(), "Reset the Device");

	return 0;
}
