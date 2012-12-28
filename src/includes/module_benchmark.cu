/**
 * @file module_benchmark.cu
 * @brief source code file for the benchmark module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <float.h>

#include <cuda.h>

#include "../3rd/HandleError.h"
#include "../3rd/sortXXL_cmd.h"

#include "constants.h"
#include "aux.h"
#include "module_file_dataset.h"
#include "module_generate_dataset.h"
#include "module_bitonic_sort.h"
#include "module_benchmark.h"

int benchmark(struct gengetopt_args_info args_info){
	int *data=NULL,
		*data_to_benchmark,
		data_allocated=FALSE,
		number_of_elements = 0,
		number_of_tests = args_info.benchmark_given == 1?args_info.benchmark_arg:1,
		current_test=0,
		i=0,
		i2=0;

	float elapsedTimes[number_of_tests], diff, sq_diff_sum;
	FILE *file;
	struct stats st;
	struct timeval start, end;
    char* aux=NULL;

	// Pick the start time
	gettimeofday(&start, NULL);

	if(args_info.input_given==1){
		// Read data set from file
		if((file_dataset(args_info, &data, &number_of_elements)==EXIT_FAILURE || number_of_elements==0)){
			printf("Data set loading failed!\n");

			return EXIT_FAILURE;
		}
		// Flag the data allocation on data set loading
		data_allocated = TRUE;
	}

	if((args_info.random_given == 1 && args_info.random_arg>0)){
		// Generate a random data set
		number_of_elements = args_info.random_arg;
		data = (int *)calloc(number_of_elements, sizeof(int));
		data_allocated = TRUE;

		// Generate a data set
		generate_dataset_from_arguments(args_info, data);
	}

	if(number_of_elements>0 && data_allocated==TRUE){
		// Capture and initialize some information data
		st.gpu_time = 0;
		st.total_sorted = number_of_elements;
		st.num_times = number_of_tests;
		st.min_time = FLT_MAX;
		st.max_time = 0;
		st.avg_time = 0;
		st.std_time = 0;

		// Allocate memory and clone the data for benchmark
		data_to_benchmark = (int *) malloc(number_of_elements*sizeof(int));
		for(current_test=0; current_test<number_of_tests; current_test++){
			memcpy(data_to_benchmark, data, number_of_elements*sizeof(int));

			printf("Running sort operation %d of %d...\n", current_test+1, number_of_tests);
//@TODO: notify the demo that the sorting is starting

			// Sort the data
			elapsedTimes[current_test] = bitonic_sort(&data_to_benchmark, number_of_elements);

			// Store some statistical data
			st.gpu_time += elapsedTimes[current_test];
			if(elapsedTimes[current_test]<st.min_time)
				st.min_time = elapsedTimes[current_test];
			if(elapsedTimes[current_test]>st.max_time)
				st.max_time = elapsedTimes[current_test];

			printf("Sort operation %d finished in %3.6f milliseconds.\n", current_test+1, elapsedTimes[current_test]);

//@TODO: notify the demo that the sort operation has finished
		}

		// Free the memory occupied by the unsorted data and update the references to the newly sorted data
		free(data);
		data = data_to_benchmark;

		// Verify the data and output the results
		for (i = 0; i < number_of_elements - 1; i++) {
			if (data[i] > data[i + 1]) {
				printf("\nSorting failed!\n");
				break;
			}
			else if (i == number_of_elements - 2){

				// Copy the first and last five elements from the sorted data
				i2=5;
				if(i2>number_of_elements){
					i2=number_of_elements;
				}
				for(current_test=0; current_test<i2; current_test++){
					st.first_five_sorted[current_test]=data[current_test];
					st.last_five_sorted[current_test]=data[number_of_elements-(i2-current_test)];
				}

				// Calculate final execution time
				gettimeofday(&end, NULL);
				st.chronological_time = time_diff(start, end)*1000;

				// Output some information
				printf("Chronological time: %3.6f\n", st.chronological_time);
				printf("GPU time: %3.6f\n", st.gpu_time);
				printf("Number of sorted elements: %d\n", st.total_sorted);

				i2=5;
				if(i2>number_of_elements){
					i2=number_of_elements;
				}
				printf("First five elements sorted: ");
				for(current_test=0; current_test<i2; current_test++){
					printf("%d%c ", st.first_five_sorted[current_test], (current_test+1>=i2)?' ':',');
					st.last_five_sorted[current_test]=data[number_of_elements-(i2-current_test)];
				}
				printf("\nLast five elements sorted: ");
				for(current_test=0; current_test<i2; current_test++){
					printf("%d%c ", st.last_five_sorted[current_test], (current_test+1>=i2)?' ':',');
				}
				printf("\n");

				// Output the benchmark specific information
				if(args_info.benchmark_given == 1){
					// Calculate the average and standard deviation (based on the algorithm http://www.strchr.com/standard_deviation_in_one_pass)
					st.avg_time = st.gpu_time/number_of_tests;
					for(current_test=0; current_test<number_of_tests; current_test++){
						diff = elapsedTimes[current_test] - st.avg_time;
						sq_diff_sum += diff * diff;
					}
					st.std_time = sq_diff_sum/number_of_tests;

					printf("Benchmark information for %d tests:\n", number_of_tests);
					printf("Minimum execution time: %3.6f\n", st.min_time);
					printf("Maximum execution time: %3.6f\n", st.max_time);
					printf("Average execution time: %3.6f\n", st.avg_time);
					printf("Standard deviation: %3.6f\n", st.std_time);
				}


				if(args_info.output_given==1){
					if((file=fopen(args_info.output_arg,"w"))==NULL){
						printf("Unable to open file %s for writing.\n", args_info.output_arg);
						exit(1);
					}else{

						// Output some information
						fprintf(file, "# Chronological time: %3.6f\n", st.chronological_time);
						fprintf(file, "# GPU time: %3.6f\n", st.gpu_time);
						fprintf(file, "# Number of sorted elements: %d\n", st.total_sorted);
						aux = get_current_time("%Y_%m_%d %H:%M:%S", strlen("@2012-12-25 15:30:00"));
						fprintf(file, "# %s",aux);
						free(aux);
						aux=NULL;

						for(i2=0; i2<number_of_elements; i2++){
							fprintf(file, "\n%d", data[i2]);
						}

						fclose(file);
					}
				}
			}
		}
	}

	if(data_allocated==TRUE){
		free(data);
	}
	return EXIT_SUCCESS;
}
