/**
 * @file module_about.cu
 * @brief source code file for the credits function
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

#include "../3rd/sortXXL_cmd.h"
#include "../3rd/HandleError.h"

#include "constants.h"
#include "module_about.h"


void about_sort_XXL(struct gengetopt_args_info args_info){
	if (args_info.about_given == 1){
    	printf("\n                           _    _  _    _ _\n                      _   \\ \\  / /\\ \\  / / |\n      ___  ___   ____| |_  \\ \\/ /  \\ \\/ /| |\n     /___)/ _ \\ / ___)  _)  )  (    )  ( | |\n    |___ | |_| | |   | |__ / /\\ \\  / /\\ \\| |_____\n    (___/ \\___/|_|    \\___)_/  \\_\\/_/  \\_\\_______)\n\n");

    	printf ("\n\t\t 2º projeto de CAD-MEICM ESTG/IPLeiria 2012-13\n");
    	printf ("\nAuthors:");
    	printf ("\nCláudio Esperança\t <2120917@my.ipleiria.pt>\nDiogo Serra\t\t <2120915@my.ipleiria.pt>\nJoão Correia\t\t <2111415@my.ipleiria.pt> \n\n");

    	system_info();
    }
}

void system_info(){
	cudaDeviceProp prop;
	int count;

	/* Get info from device */
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
