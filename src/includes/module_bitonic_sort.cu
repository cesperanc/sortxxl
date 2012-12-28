/**
 * @file module_bitonic_sort.cu
 * @brief source code file for the bitonic sort algorithm implementation
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../3rd/sortXXL_cmd.h"
#include "../3rd/HandleError.h"

#include "constants.h"
#include "module_bitonic_sort.h"


/**
 * @brief Define the maximum number of threads for the sorting operation
 */
#define MAX_THREADS 128

// n limited to 2^22=4194304
float bitonic_sort(int **values, int n){
	size_t size = n * sizeof(int);
	int *dev_values, *resized_values, k, j, padded_size=n;
	cudaEvent_t start, stop;
    float elapsedTime;

	resized_values = *values;

	// Add padding to align the number of data elements to a power of two
	padded_size = get_next_power_of_two(n);
	if(padded_size>n){
		// Allocate more memory for the padding
		resized_values = (int*) realloc(*values, padded_size*sizeof(int));
		if (resized_values!=NULL) {

			// Align the data
			pad_data_to_align(resized_values, n, padded_size);

			// Update the size
			size = padded_size*sizeof(int);
		}else{
			printf("Problem while allocating the necessary memory for padding (memory exhausted?)\n");
		}
	}

	// allocate device memory
	HANDLE_ERROR( cudaMalloc((void**)&dev_values, size) );

	// copy data to device
	HANDLE_ERROR( cudaMemcpy(dev_values, resized_values, size, cudaMemcpyHostToDevice) );

	printf("Beginning kernel execution...\n");

	// Create the timers
	HANDLE_ERROR (cudaEventCreate (&start));
	HANDLE_ERROR (cudaEventCreate (&stop));

	// Synchronize the threads
	HANDLE_ERROR( cudaThreadSynchronize() );

	// Start the timer
	HANDLE_ERROR (cudaEventRecord (start, 0));

	// execute kernel
	for (k = 2; k <= padded_size; k <<= 1) {
		for (j = k >> 1; j > 0; j = j >> 1) {
			if (padded_size < MAX_THREADS)
				cuda_bitonic_sort <<< 1, padded_size >>> (dev_values, j, k, padded_size);
			else
				cuda_bitonic_sort <<< padded_size / MAX_THREADS, MAX_THREADS >>> (dev_values, j, k, padded_size);
		}
	}

	// Wait for synchronization of all threads
	HANDLE_ERROR( cudaThreadSynchronize() );

	/* Terminate the timer */
	HANDLE_ERROR (cudaEventRecord (stop, 0));
	HANDLE_ERROR (cudaEventSynchronize (stop));
	HANDLE_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	HANDLE_ERROR (cudaEventDestroy (start));
	HANDLE_ERROR (cudaEventDestroy (stop));

	// Copy data back to host
	HANDLE_ERROR( cudaMemcpy(resized_values, dev_values, size, cudaMemcpyDeviceToHost) );

	// Update with the sorted values
	*values = resized_values;

	// Remove the padding and free some memory
	if(padded_size>n){
		padded_size=n;
		// Allocate more memory for the padding
		resized_values = (int*) realloc(*values, padded_size*sizeof(int));
		if (resized_values!=NULL) {
			// Update the size
			size = padded_size*sizeof(int);

			// Update the reference
			*values = resized_values;
		}
	}

	// free memory
	HANDLE_ERROR( cudaFree(dev_values) );

	// Free device resources (replaces depreciated cudaThreadExit)
	cudaDeviceReset();

	// Return the elapsed time
	return elapsedTime;
}

// Kernel function
__global__ void cuda_bitonic_sort(int* values, int j, int k, int n) {
	const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		int ixj = idx^j;
 		if (ixj > idx) {
			if ((idx&k) == 0 && values[idx] > values[ixj]) {
				//exchange(idx, ixj);
				int tmp = values[idx];
				values[idx] = values[ixj];
				values[ixj] = tmp;
			}
			if ((idx&k) != 0 && values[idx] < values[ixj]) {
				//exchange(idx, ixj);
				int tmp = values[idx];
				values[idx] = values[ixj];
				values[ixj] = tmp;
			}
		}
	}
}

int get_next_power_of_two(int size){
	int n = 1;
	while (n < size) n *= 2;
	return n;
}

void pad_data_to_align_with(int *data, int current_size, int to_size, int with){
	int i;
	for(i=current_size; i<to_size; i++){
		data[i] = with;
	}
}

void pad_data_to_align(int *data, int current_size, int to_size){
	pad_data_to_align_with(data, current_size, to_size, INT_MAX);
}

int pad_data_to_align_with_next_power_of_two_with(int *data, int current_size, int with){
	int next_power_of_two = get_next_power_of_two(current_size);
	pad_data_to_align_with(data, current_size, next_power_of_two, with);
	return next_power_of_two;
}

int pad_data_to_align_with_next_power_of_two_with(int *data, int current_size){
	int next_power_of_two = get_next_power_of_two(current_size);
	pad_data_to_align(data, current_size, next_power_of_two);
	return next_power_of_two;
}
