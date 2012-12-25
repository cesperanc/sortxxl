#include <stdio.h>
#include <cuda.h>
#include "HandleError.h"
#include "module_sortXXL.h"

/* Kernel */
__global__ void DefaultKernel(float *a, float *b, float *c, int N)
{
	/* Compute total number of threads per grid */
	int TotalThreads = (blockDim.x*blockDim.y*blockDim.z) * (gridDim.x*gridDim.y*gridDim.z);
	/// as seen in http://martinpeniak.com/ on 2012-11-05
	// unique block index inside a 3D block grid
	int blockId = blockIdx.x //1D
			+ blockIdx.y * gridDim.x //2D
			+ gridDim.x * gridDim.y * blockIdx.z; //3D
	/// global unique thread index for a 3D block dimension
	int tid =  blockId * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y)) 
			+ (threadIdx.y * blockDim.x) 
			+ threadIdx.x; 
	while (tid < N){
		c[tid]=a[tid] + b[tid];
		tid = tid+TotalThreads;
	}
	__syncthreads();
}

int main (int argc, char *argv[]){

	int N = 32768;
	float a[N], b[N], c[N];
	float *a_dev, *b_dev, *c_dev;

	/*Allocate the memory on the cpu*/
	HANDLE_ERROR (cudaMalloc((void**)&a_dev, N * sizeof(float)));
	HANDLE_ERROR (cudaMalloc((void**)&b_dev, N * sizeof(float)));
	HANDLE_ERROR (cudaMalloc((void**)&c_dev, N * sizeof(float)));
	/* Initialize the arrays 'a' and 'b' on the CPU */
	for (int i=0; i<N; i++){
		a[i] = i;
		b[i] = i;
	}
	/* Copy 'a' and 'b' to GPU's memory – cudaMemcpy(.) */
	HANDLE_ERROR (cudaMemcpy(a_dev,a,N*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR (cudaMemcpy(b_dev,b,N*sizeof(float),cudaMemcpyHostToDevice));
	/* Inform user */
	printf("[INFO] Calling: a multidimensional kernel (DefaultKernel)");
	dim3 B(1,1,1); dim3 T(1,1,1); DefaultKernel<<<B,T>>>(a_dev,b_dev,c_dev,N);
	/* Copy results 'c_dev' to CPU's 'c' */
	HANDLE_ERROR (cudaMemcpy(c, c_dev, N*sizeof(float),cudaMemcpyDeviceToHost));
		printf("\n");
	for(int i=0;i<N;i++){
		printf("[%d]: %2.2f + %2.2f = %2.2f // %2.2f<-GPU result\n",i, a[i],b[i],a[i]+b[i],c[i]);
	}
	/* Get info from device */
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	for (int i=0; i< count; i++){
		HANDLE_ERROR (cudaGetDeviceProperties(&prop,i));
		printf( " --- Information for device %d ---\n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Number of cores: %s\n", number_of_cores(prop.major, prop.minor) );
		printf( " --- Memory Information for device %d ---\n", i );
		printf( "Total global mem: %zu\n",prop.totalGlobalMem );
		printf( " --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count: %d\n",prop.multiProcessorCount );
	}
	/* Free device memory */
	HANDLE_ERROR (cudaFree(a_dev));
	HANDLE_ERROR (cudaFree(b_dev));
	HANDLE_ERROR (cudaFree(c_dev));
	
	return 0;
}
char* number_of_cores(double gpu_major, double gpu_minor){
	double cap = gpu_major*10+gpu_minor;
	if(cap <= 13){
		return "8";
	}else if(cap == 20){
		return "32";
	}else if(cap == 21){
		return "48";
	}else if(cap == 30){
		return "192";
	}else{
		return "unsupported";
	}
}
