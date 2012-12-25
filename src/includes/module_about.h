/**
 * @file module_about.h
 * @brief header file for the credits module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_CREDITS_H
#define	MODULE_CREDITS_H

/**
 * @brief Output the credits message
 * @param args_info struct gengetopt_args_info with the parameters given to the application
 *
 * @author Joao Correia <joao.pedro.j.correia@gmail.com>
 */
void about_sort_XXL(struct gengetopt_args_info args_info);

/**
 * @brief Output the CUDA system information
 * @author Diogo Serra <2120915@my.ipleiria.pt>
 */
void system_info();

/**
 * @brief Compute the number of cores string
 * @author Diogo Serra <2120915@my.ipleiria.pt>
 */
char* number_of_cores(double, double);

#endif	/* MODULE_CREDITS_H */

