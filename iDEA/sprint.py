######################################################################################
# Name: SPRINT                                                                       #
######################################################################################
# Author(s): Jack Wetherell, Leopold Talirz                                          #
######################################################################################
# Description:                                                                       #
# Prints text to screen overwriting last line                                        #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################
from __future__ import print_function

# Do not run stand-alone
if(__name__ == '__main__'):
    print('do not run stand-alone')
    quit()

# Library imports


priority_dict = {
  'low': 2,
  'default': 1,
  'high': 0}

# Sprint function
def sprint(string, priority=1, verbosity='default', newline=True):
    """Customized print function

    parameters
    ----------
    string : string
        string to be printed
    priority: int
        priority of message, possible values are
        0: debug
        1: normal
        2: important
    msglevel: string
        'debug' print all messages
        'high'  print messages with priority >= 1
        'low'   print messages with priority >= 2
    newline : bool
        If False, overwrite the last line
    """
    if priority >= priority_dict[verbosity]:
        if newline:
            print(string)
        else:
            #print('\r' + string, end='')
            print(string, end='\r')

#import sys
#
#    
#    f(n == verbosity):
#       if(s == 1):
#           sys.stdout.write('\033[K')
#           sys.stdout.flush()
#           sys.stdout.write('\r' + text)
#           sys.stdout.flush()
#       else:
#           print(text)
#
