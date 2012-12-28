################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../extras/genRandom.c \
../extras/regex.c 

CU_SRCS += \
../extras/bitonic_sort.cu \
../extras/quickSort.cu \
../extras/radix_sort.cu 

C_DEPS += \
./extras/genRandom.d \
./extras/regex.d 


# Each subdirectory must supply rules for building sources it contributes
extras/%.o: ../extras/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -O0 -g -c -Xcompiler -fmessage-length=0 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

extras/%.o: ../extras/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


