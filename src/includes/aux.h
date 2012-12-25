/**
 * @file aux.h
 * @brief header file for the auxiliary functions
 * @date 2012-10-22 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef AUX_H_
#define AUX_H_

// defines
/**
 * @brief Macro to print on the stderr useful depuration information.
 * It accepts a variable number of parameters
 *
 * @see my_debug() for more information about this function
 */
#define MY_DEBUG(...) my_debug(__FILE__, __LINE__,__VA_ARGS__)

// prototypes
void my_debug(const char*, const int, char*, ...);

int directory_exists(char*);
int file_exists(char*, char*);

FILE* open_stdout_file(char*, char*);
void close_stdout_file(FILE*);
int make_directory(char*, mode_t);
int remove_directory(const char *);
char* get_current_time(char*, int);
char* concatenate_filename(const char*, const char*, const char);
char* base_name (char *, const char);

#endif /* AUX_H_ */
