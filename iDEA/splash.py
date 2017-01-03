"""Prints iDEA logo as splash
"""


import info
import textwrap


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
