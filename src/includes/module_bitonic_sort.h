/**
 * @file module_bitonic_sort.h
 * @brief header file for the bitonic sort algorithm
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_BITONIC_SORT_H
#define	MODULE_BITONIC_SORT_H

/**
 * @brief Sort an array of integers using the bitonic algorithm
 * @param values with the pointer for the address to starting ordering the data
 * @param n with the number of elements in the array
 * @return float with the elapsed time in the sort process
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
float bitonic_sort(int **values, int n);

/**
 * @brief Generates a random data set of integers in a specific interval
 * @param values with the pointer for the address were to store the data
 * @param j
 * @param k
 * @param n with the number of elements in the array
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
__global__ static void cuda_bitonic_sort(int* values, int j, int k, int n);

/**
 * @brief Get the next power of two greater than size
 * @param size with the current size
 * @return integer with the next value
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
int get_next_power_of_two(int size);

/**
 * @brief Align the data with a specific size using a padding
 * @param data with the array to store the data
 * @param current_size with the current size of the array without padding
 * @param to_size with final size of the array
 * @param with with the value to use as a padding
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
void pad_data_to_align_with(int *data, int current_size, int to_size, int with);

/**
 * @brief Align the data with a specific size using 0 as padding
 * @param data with the array to store the data
 * @param current_size with the current size of the array without padding
 * @param to_size with final size of the array
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
void pad_data_to_align(int *data, int current_size, int to_size);

/**
 * @brief Align the data with a power of two size using a padding
 * @param data with the array to store the data
 * @param current_size with the current size of the array without padding
 * @param with with the value to use as a padding
 * @return integer with the final array size
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
int pad_data_to_align_with_next_power_of_two_with(int *data, int current_size, int with);

/**
 * @brief Align the data with a power of two size using 0 as padding
 * @param data with the array to store the data
 * @param current_size with the current size of the array without padding
 * @return integer with the final array size
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
int pad_data_to_align_with_next_power_of_two_with(int *data, int current_size);

#endif	/* MODULE_BITONIC_SORT_H */

