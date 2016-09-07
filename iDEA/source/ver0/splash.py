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
   sprint.sprint('  |        Created by Piers Lillystone, Matt Hodgson         |',1,0,msglvl)                         
   sprint.sprint('  |                    and Jack Wetherell                    |',1,0,msglvl)
   sprint.sprint('  |                                                          |',1,0,msglvl)
   sprint.sprint('  |          With contribution from: Aaron Long              |',1,0,msglvl)
   sprint.sprint('  |                                  Mike Entwistle          |',1,0,msglvl)
   sprint.sprint('  |                                  James Ramsden           |',1,0,msglvl)
   sprint.sprint('  |                                  Thomas Durrant          |',1,0,msglvl)
   sprint.sprint('  |                                  Jacob Chapman           |',1,0,msglvl)
   sprint.sprint('  |                                  Matthew Smith           |',1,0,msglvl)
   sprint.sprint('  |                                                          |',1,0,msglvl)
   sprint.sprint('  |                    University of York                    |',1,0,msglvl)
   sprint.sprint('  +----------------------------------------------------------+',1,0,msglvl)
   sprint.sprint('                                                              ',1,0,msglvl)
