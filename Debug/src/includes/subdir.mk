################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/includes/aux.cu \
../src/includes/module_about.cu \
../src/includes/module_benchmark.cu \
../src/includes/module_bitonic_sort.cu \
../src/includes/module_file_dataset.cu \
../src/includes/module_generate_dataset.cu \
../src/includes/module_system_info.cu 


# Each subdirectory must supply rules for building sources it contributes
src/includes/%.o: ../src/includes/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O0 -g -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


