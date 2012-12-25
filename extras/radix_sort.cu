/*
 *
 * radix_sort.cu
 *
 */
 
 #include <time.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <cutil_inline.h>

 #define MAX_THREADS	128
 #define N		512

 int* r_values;
 int* d_values;
 int* t_values;

 int* d_split;
 int* d_e;
 int* d_f;
 int* d_t;

 // convert to binary string
 char* Itoa(int value, char* str, int radix) {
        static char dig[] =
                "0123456789"
                "abcdefghijklmnopqrstuvwxyz";
        int n = 0, neg = 0;
        unsigned int v;
        char* p, *q;
        char c;

        if (radix == 10 && value < 0) {
                value = -value;
                neg = 1;
        }
        v = value;
        do {
                str[n++] = dig[v%radix];
                v /= radix;
        } while (v);
        if (neg)
                str[n++] = '-';
        str[n] = '\0';

        for (p = str, q = p + (n-1); p < q; ++p, --q)
        c = *p, *p = *q, *q = c;
        return str;
}

 // initialize data set
 void Init(int* values, int i) {
        srand( time(NULL) );
        printf("\n------------------------------\n");

        if (i == 0) {
        // Uniform distribution
                printf("Data set distribution: Uniform\n");
                for (int x = 0; x < N; ++x) {
                        values[x] = rand() % 100;
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 1) {
        // Gaussian distribution
        #define MEAN    100
        #define STD_DEV 5
                printf("Data set distribution: Gaussian\n");
                float r;
                for (int x = 0; x < N; ++x) {
                        r  = (rand()%3 - 1) + (rand()%3 - 1) + (rand()%3 - 1);
                        values[x] = int( round(r * STD_DEV + MEAN) );
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 2) {
        // Bucket distribution
                printf("Data set distribution: Bucket\n");
                int j = 0;
                for (int x = 0; x < N; ++x, ++j) {
                        if (j / 20 < 1)
                                values[x] = rand() % 20;
                        else if (j / 20 < 2)
                                values[x] = rand() % 20 + 20;
                        else if (j / 20 < 3)
                                values[x] = rand() % 20 + 40;
                        else if (j / 20 < 4)
                                values[x] = rand() % 20 + 60;
                        else if (j / 20 < 5)
                                values[x] = rand() % 20 + 80;
                        if (j == 100)
                                j = 0;
                        //printf("%d ", values[x]);
                }
        }
        else if (i == 3) {
        // Sorted distribution
                printf("Data set distribution: Sorted\n");
                /*for (int x = 0; x < N; ++x)
                        printf("%d ", values[x]);
                */
        }
	else if (i == 4) {
        	// Zero distribution
                printf("Data set distribution: Zero\n");
	        int r = rand() % 100;
         	for (int x = 0; x < N; ++x) {
                       	values[x] = r;
                       	//printf("%d ", values[x]);
	        }
        }

	// Convert to binary
        char binary_string[8];
	for (int x = 0; x < N; ++x) {
        	Itoa(r_values[x], binary_string, 2);		// convert to binary string
                r_values[x] = atoi(binary_string);		// convert to binary int
		//printf(" %d\n", r_values[x]);
	}

       	printf("\n");
}

 // Kernel function
 __global__ static void Radix_sort(int* values, int* temp, int loop, int* split, int* e, int* f, int* t) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int remainder[N], quotient[N];
	int f_count, totalFalses;

	if (idx < N) {
		// split based on least significant bit
		quotient[idx] = values[idx];
		for (int x = 0; x < loop + 1; ++x) {
			remainder[idx] = quotient[idx] % 10;
			quotient[idx] = quotient[idx] / 10;
		}

		// set e[idx] = 0 in each 1 input and e[idx] = 1 in each 0 input	
		if (remainder[idx] == 1) {
			split[idx] = 1;
			e[idx] = 0;
		}
		else {
			split[idx] = 0;
			e[idx] = 1;
		}
	}
	__syncthreads();

	if (idx < N) {
		// scan the 1s
		f_count = 0;
		for (int x = 0; x < N; ++x) {
			f[x] = f_count;
			if (e[x] == 1)
				f_count++;
		}

		// calculate totalFalses
		totalFalses = e[N-1] + f[N-1];

		if (split[idx] == 1) {
			// t = idx - f + totalFalses
			t[idx] = idx - f[idx] + totalFalses;
		}
		else if (split[idx] == 0) {
			// t = f[idx]
			t[idx] = f[idx];
		}

		// Scatter input using t as scatter address
		temp[t[idx]] = values[idx];
	}
	__syncthreads();
	
	// copy new arrangement back to values
	if (idx < N) {
		values[idx] = temp[idx];
	}
}

 // program main
 int main(int argc, char** argv) {
	printf("./radix_sort starting with %d numbers...\n", N);
	unsigned int hTimer;
	size_t size = N * sizeof(int);

	// allocate host memory
	r_values = (int*)malloc(size);

	// allocate device memory
	cutilSafeCall( cudaMalloc((void**)&d_values, size) );
	cutilSafeCall( cudaMalloc((void**)&t_values, size) );
	cutilSafeCall( cudaMalloc((void**)&d_split, size) );
	cutilSafeCall( cudaMalloc((void**)&d_e, size) );
	cutilSafeCall( cudaMalloc((void**)&d_f, size) );
	cutilSafeCall( cudaMalloc((void**)&d_t, size) );
	
	/* Types of data sets to be sorted:
	 *	1. Normal distribution
	 *	2. Gaussian distribution
	 *	3. Bucket distribution
	 * 	4. Sorted distribution
	 *	5. Zero distribution
	 */

	for (int i = 0; i < 5; ++i) {
		// Initialize data set
		Init(r_values, i);

		// copy data to device
		cutilSafeCall( cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice) );
		
		printf("Beginning kernel execution...\n");

		cutilCheckError( cutCreateTimer(&hTimer) );
		cutilSafeCall( cudaThreadSynchronize() );
		cutilCheckError( cutResetTimer(hTimer) );
		cutilCheckError( cutStartTimer(hTimer) );

		// execute kernel
		for (int j = 0; j < 8; ++j) {
			Radix_sort <<< 1, N >>> (d_values, t_values, j, d_split, d_e, d_f, d_t);
		}
		cutilCheckMsg( "Kernel execution failed...\n" );

		cutilSafeCall( cudaThreadSynchronize() );
		cutilCheckError( cutStopTimer(hTimer) );
		double gpuTime = cutGetTimerValue(hTimer);

		printf("\nKernel execution completed in %f ms\n", gpuTime);

		// copy data back to host
		cutilSafeCall( cudaMemcpy(r_values, t_values, size, cudaMemcpyDeviceToHost) );

		// convert to decimal & test print
		int r;
		for (int x = 0; x < N; ++x) {
			int 	val = 0,
				power = 0;
			while (r_values[x] > 0) {
				r = r_values[x] % 10;
				val = val + int(pow(2.0, power) * r);
				r_values[x] = r_values[x] / 10;
				power++;
			}
			r_values[x] = val;
			//printf("%d ", r_values[x]);
		}
		//printf("\n");

		// test
		printf("\nTesting results...\n");
		for (int x = 0; x < N - 1; ++x) {
			if (r_values[x] > r_values[x + 1]) {
				printf("Sorting failed.\n");
				break;
			}
			else
				if (x == N - 2)
					printf("SORTING SUCCESSFUL\n");
		}
	}

	// free memory
	cutilSafeCall( cudaFree(d_values) );
	cutilSafeCall( cudaFree(t_values) );
	cutilSafeCall( cudaFree(d_split) );
	cutilSafeCall( cudaFree(d_e) );
	cutilSafeCall( cudaFree(d_f) );
	cutilSafeCall( cudaFree(d_t) );
	free(r_values);

	cutilExit(argc, argv);
	cudaThreadExit();
 }
