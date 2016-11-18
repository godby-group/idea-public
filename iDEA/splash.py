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
import info
import textwrap

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
   sprint.sprint('  |{:^58}|'.format('Version {}'.format(info.version)))
   sprint.sprint('  |                                                          |',1,verbosity)
   lines = textwrap.wrap('Created by ' + info.authors_long, width=45)
   for l in lines:
      sprint.sprint('  |{:^58}|'.format(l))
   sprint.sprint('  |                                                          |',1,verbosity)
   sprint.sprint('  |                    University of York                    |',1,verbosity)
   sprint.sprint('  +----------------------------------------------------------+',1,verbosity)
   sprint.sprint('                                                              ',1,verbosity)
