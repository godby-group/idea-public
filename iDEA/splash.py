"""Prints iDEA logo as splash
"""
from __future__ import absolute_import

from . import info
import textwrap


def draw(pm):
   pm.sprint('                                                              ')
   pm.sprint('                *    ****     *****       *                   ')
   pm.sprint('                     *   *    *          * *                  ')
   pm.sprint('                *    *    *   *         *   *                 ')
   pm.sprint('                *    *     *  *****    *     *                ')
   pm.sprint('                *    *    *   *       *********               ')
   pm.sprint('                *    *   *    *      *         *              ')
   pm.sprint('                *    ****     ***** *           *             ')
   pm.sprint('                                                              ')
   pm.sprint('  +----------------------------------------------------------+')
   pm.sprint('  |          Interacting Dynamic Electrons Approach          |')
   pm.sprint('  |              to Many-Body Quantum Mechanics              |')
   pm.sprint('  |                                                          |')
   pm.sprint('  |{:^58}|'.format('Public Release {}'.format(info.release)))
   pm.sprint('  |                                                          |')
   sha1 = info.get_sha1()
   if sha1 != "Unknown":
       pm.sprint('  |{:^58}|'.format('git commit hash'))
       pm.sprint('  |{:^58}|'.format(sha1))
       pm.sprint('  |{:^58}|'.format(''))
   lines = textwrap.wrap('Created by ' + info.authors_long, width=45)
   for l in lines:
      pm.sprint('  |{:^58}|'.format(l))
   pm.sprint('  |                                                          |')
   pm.sprint('  |                    University of York                    |')
   pm.sprint('  +----------------------------------------------------------+')
   pm.sprint('                                                              ')
