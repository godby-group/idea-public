######################################################################################
# Name: SPRINT                                                                       #
######################################################################################
# Author(s): Jack Wetherell                                                          #
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

# Do not run stand-alone
if(__name__ == '__main__'):
    print('do not run stand-alone')
    quit()

# Library imports
import sys

# Sprint function
def sprint(text, n, s, msglvl):
    if(n == msglvl):
	if(s == 1):
	    sys.stdout.write('\033[K')
	    sys.stdout.flush()
	    sys.stdout.write('\r' + text)
	    sys.stdout.flush()
        else:
            print(text)

