/**
 * @file constants.h
 * @brief header file for constants
 * @date 2012-10-22 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef SORTXXL_CONSTANTS_H_
#define SORTXXL_CONSTANTS_H_

/**
 * A default type for a boolean false value
 */
#define FALSE 0

/**
 * A default type for a boolean true value
 */
#define TRUE 1

// exit constants
/**
 * Define the default value for the OK message
 */
#define M_OK 0
/**
 * Define the exit value for the invalid parameters message
 */
#define M_INVALID_PARAMETERS 1

/**
 * Define the exit value for the make path fail message
 */
#define M_FAILED_MK_PATH 2

/**
 * Define the exit value for the remove directory fail message
 */
#define M_FAILED_REMOVE_DIRECTORY 3

/**
 * Define the exit value for the memory allocation fail message
 */
#define M_FAILED_MEMORY_ALLOCATION 4

/**
 * Define the exit value for the localtime fail message
 */
#define M_LOCALTIME_FAILED 5

/**
 * Define the exit value for the format time fail message
 */
#define M_FORMATTIME_FAILED 6

/**
 * Define the exit value for the open_stdout_file fail message
 */
#define M_OPEN_STDOUT_FILE_FAILED 7

/**
 * Define the exit value for the output_directory fail message
 */
#define M_OUTPUT_DIRECTORY_FAILED 8

/**
 * Define the exit value for the strcpy fail message
 */
#define M_FAILED_STRCPY 9

#endif /* SORTXXL_CONSTANTS_H_ */
