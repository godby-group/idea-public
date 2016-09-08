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
def draw(msglvl):
   sprint.sprint('                                                              ',1,0,msglvl)
   sprint.sprint('                  *    ****     *****       *                 ',1,0,msglvl)
   sprint.sprint('                       *   *    *          * *                ',1,0,msglvl)
   sprint.sprint('                  *    *    *   *         *   *               ',1,0,msglvl)
   sprint.sprint('                  *    *     *  *****    *     *              ',1,0,msglvl)
   sprint.sprint('                  *    *    *   *       *********             ',1,0,msglvl)
   sprint.sprint('                  *    *   *    *      *         *            ',1,0,msglvl)
   sprint.sprint('                  *    ****     ***** *           *           ',1,0,msglvl)
   sprint.sprint('                                                              ',1,0,msglvl)
   sprint.sprint('  +----------------------------------------------------------+',1,0,msglvl)
   sprint.sprint('  |           Interacting Dynamic Electrons Approach         |',1,0,msglvl)
   sprint.sprint('  |              to Many-Body Quantum Mechanics              |',1,0,msglvl)
   sprint.sprint('  |                                                          |',1,0,msglvl)
   sprint.sprint('  |        Created by Piers Lillystone, James Ramsden        |',1,0,msglvl)                         
   sprint.sprint('  |        Matt Hodgson, Thomas Durrant, Jacob Chapman       |',1,0,msglvl)
   sprint.sprint('  |      Thomas Durrant, Jack Wetherell, Matthew Smith       |',1,0,msglvl)
   sprint.sprint('  |                Mike Entwistle and Aaron Long             |',1,0,msglvl)
   sprint.sprint('  |                                                          |',1,0,msglvl)
   sprint.sprint('  |                    University of York                    |',1,0,msglvl)
   sprint.sprint('  +----------------------------------------------------------+',1,0,msglvl)
   sprint.sprint('                                                              ',1,0,msglvl)
