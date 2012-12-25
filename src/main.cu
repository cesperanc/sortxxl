/**
* \mainpage
* The sortXXL is an application to sort a big amount of numbers
* 
* 
* @file main.c
* @brief Main source file for the sortXLL program
* @date 2012-12-03 File creation
* @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "3rd/debug.h"
#include "3rd/sortXXL_cmd.h"

#include "includes/constants.h"
#include "includes/aux.h"

#include "includes/module_about.h"

#include "main.h"

/**
 * @brief The main program function
 * @param argc integer with the number of command line options
 * @param argv *char[] with the command line options
 * @return integer 0 on a successfully exit, another integer value otherwise
 *
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
int main(int argc, char *argv[]){
    
    /* Variable declarations */
    struct gengetopt_args_info args_info;// structure for the command line parameters processing
    int result = EXIT_SUCCESS;
    
    // Initializes the command line parser and check for the application parameters
    if (cmdline_parser(argc,argv,&args_info) != 0){
        DEBUG("\nInvalid parameters");
        result = M_INVALID_PARAMETERS;
    }
    
   //if (args_info.about_given < 1 && args_info.help_given < 1){
    
        // Output directory manager. Use the args_info.dir_arg afterwards as the base directory for the files
        /*if(output_directory(&args_info)!=M_OK){
            ERROR(M_OUTPUT_DIRECTORY_FAILED,"\nUnable to use the output directory provided\n");
        }*/

        // Outputs the main file
        //output_kernel(args_info);
        
        // Create the makefile file
        //ouput_makefile(args_info);
    //}
    
    // Output the credits
    about_sort_XXL(args_info);
    
    // Free the command line parser memory
    cmdline_parser_free(&args_info);

    return result;
}
