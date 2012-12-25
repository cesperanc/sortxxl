/**
 * @file module_credits.c
 * @brief source code file for the credits function
 * @date 2012-11-08 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>

#include "../3rd/sortXXL_cmd.h"

#include "constants.h"
#include "module_credits.h"

/**
 * @brief Output the credits message
 * @param args_info struct gengetopt_args_info with the parameters given to the application
 *
 * @author Joao Correia <joao.pedro.j.correia@gmail.com>
 */
void credits(struct gengetopt_args_info args_info){
    if (args_info.about_given == 1){
    	printf("\n\t\t   ____          _        ____            \n");
        printf("\t\t  / ___|   _  __| | __ _ / ___| ___ _ __  \n");
        printf("\t\t | |  | | | |/ _\' |/ _\' | |  _ / _ \\ \'_ \\ \n");
        printf("\t\t | |__| |_| | (_| | (_| | |_| |  __/ | | |\n");
        printf("\t\t  \\____\\__,_|\\__,_|\\__,_|\\____|\\___|_| |_|\n");
    
    	printf ("\t\t MEI-CM 2012/2013 (Computação de Alto Desempenho)\n");
    	printf ("Authors:\n");
    	printf ("Cláudio Esperança\t <2120917@my.ipleiria.pt>\nDiogo Serra\t\t <2120915@my.ipleiria.pt>\nJoão Correia\t\t <2111415@my.ipleiria.pt> \n\n");
    }
}
