/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/

/********************************************************************
*	By: Khaled Alshatti												*
*	Project 1														*
*	Date: 06/6/2019													*
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
__global__ void PDH_baseline(bucket *histogram, atom * atomList, double width, int size) {
	int i, j, x, h_pos;
    double dist;
    
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = i + 1;
	
    for(x = j; x < size; x++) 
    {
		dist = p2p_distance(atomList, i, x);
		h_pos = (int) (dist / width);
		atomicAdd(&histogram[h_pos].d_cnt, 1);
	}
	__syncthreads();
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	SDH solution in GPU thread 
*/
__global__ void PDH2D_baseline(bucket *histogram, atom * atomList, double width) 
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

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

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
	atom *deviceAtomList = NULL;

	//Check for cuda error
	checkErrorCuda(cudaMalloc((void**) &deviceHistogram, histogramSize), "Cuda Malloc Historgram");
	checkErrorCuda(cudaMalloc((void**) &deviceAtomList, atomSize), "Cuda Malloc Atom List");
	checkErrorCuda(cudaMemcpy(deviceHistogram, histogram, histogramSize, cudaMemcpyHostToDevice), "Copy Histogrm to Device");
	checkErrorCuda(cudaMemcpy(deviceAtomList, atom_list, atomSize, cudaMemcpyHostToDevice), "Copy Atom List to Device");

	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* calling cuda kernel */
	PDH_baseline <<<ceil(PDH_acnt/256.0), 256>>> (deviceHistogram, deviceAtomList, PDH_res, PDH_acnt);
	//PDH2D_baseline <<<ceil(PDH_acnt/256.0), 256>>> (deviceHistogram, deviceAtomList, PDH_res);

	checkErrorCuda(cudaGetLastError(), "Check last error");
	checkErrorCuda(cudaMemcpy(histogram, deviceHistogram, histogramSize, cudaMemcpyDeviceToHost), "Copy Device Histogram");

	/* check the total running time */ 
	report_running_time();
	
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


