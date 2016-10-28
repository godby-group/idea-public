######################################################################################
# Name: SPLASH                                                                       #
######################################################################################
# Author(s): Jack Wetherell                                                          #
######################################################################################
# Description:                                                                       #
# Prints the splash                                                                  #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Do not run stand-alone
if(__name__ == '__main__'):
    print('do not run stand-alone')
    quit()

# Library imports
import sprint

# Draw splash to screen
def draw(verbosity):
   sprint.sprint('                                                              ',1,verbosity)
   sprint.sprint('                  *    ****     *****       *                 ',1,verbosity)
   sprint.sprint('                       *   *    *          * *                ',1,verbosity)
   sprint.sprint('                  *    *    *   *         *   *               ',1,verbosity)
   sprint.sprint('                  *    *     *  *****    *     *              ',1,verbosity)
   sprint.sprint('                  *    *    *   *       *********             ',1,verbosity)
   sprint.sprint('                  *    *   *    *      *         *            ',1,verbosity)
   sprint.sprint('                  *    ****     ***** *           *           ',1,verbosity)
   sprint.sprint('                                                              ',1,verbosity)
   sprint.sprint('  +----------------------------------------------------------+',1,verbosity)
   sprint.sprint('  |           Interacting Dynamic Electrons Approach         |',1,verbosity)
   sprint.sprint('  |              to Many-Body Quantum Mechanics              |',1,verbosity)
   sprint.sprint('  |                                                          |',1,verbosity)
   sprint.sprint('  |        Created by Piers Lillystone, James Ramsden        |',1,verbosity)                         
   sprint.sprint('  |        Matt Hodgson, Thomas Durrant, Jacob Chapman       |',1,verbosity)
   sprint.sprint('  |      Thomas Durrant, Jack Wetherell, Matthew Smith       |',1,verbosity)
   sprint.sprint('  |                Mike Entwistle and Aaron Long             |',1,verbosity)
   sprint.sprint('  |                                                          |',1,verbosity)
   sprint.sprint('  |                    University of York                    |',1,verbosity)
   sprint.sprint('  +----------------------------------------------------------+',1,verbosity)
   sprint.sprint('                                                              ',1,verbosity)
