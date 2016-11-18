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
import info
import textwrap

# Draw splash to screen
def draw(pm):
   pm.sprint('                                                              ',1)
   pm.sprint('                  *    ****     *****       *                 ',1)
   pm.sprint('                       *   *    *          * *                ',1)
   pm.sprint('                  *    *    *   *         *   *               ',1)
   pm.sprint('                  *    *     *  *****    *     *              ',1)
   pm.sprint('                  *    *    *   *       *********             ',1)
   pm.sprint('                  *    *   *    *      *         *            ',1)
   pm.sprint('                  *    ****     ***** *           *           ',1)
   pm.sprint('                                                              ',1)
   pm.sprint('  +----------------------------------------------------------+',1)
   pm.sprint('  |           Interacting Dynamic Electrons Approach         |',1)
   pm.sprint('  |              to Many-Body Quantum Mechanics              |',1)
   pm.sprint('  |                                                          |',1)
   pm.sprint('  |{:^58}|'.format('Version {}'.format(info.version)))
   pm.sprint('  |                                                          |',1)
   lines = textwrap.wrap('Created by ' + info.authors_long, width=45)
   for l in lines:
      pm.sprint('  |{:^58}|'.format(l))
   pm.sprint('  |                                                          |',1)
   pm.sprint('  |                    University of York                    |',1)
   pm.sprint('  +----------------------------------------------------------+',1)
   pm.sprint('                                                              ',1)
