######################################################################################
# Name: RUN                                                                          #
######################################################################################
# Author(s): Jack Wetherell                                                          #
######################################################################################
# Description:                                                                       #
# Runs jobs requested in parameters                                                  #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Library imports
import os
import splash
import sprint
import SPiDEA
import iDEA_MB2
import iDEA_MB3
import parameters as pm

# Draw splash to screen
splash.draw(pm.msglvl)

# Execute required jobs
if(pm.NE==1):
   if(pm.MB == 1):
      SPiDEA.main()
   if(pm.REV == 1):
      import iDEA_RE
      iDEA_RE.main()
   if(pm.LDA == 1):
      import iDEA_LDA1
   if(pm.MLP == 1):
      print('MLP: not yet implemented')
if(pm.NE==2):
   if(pm.MB == 1):
      iDEA_MB2.main()
   if(pm.REV == 1):
      import iDEA_RE
      iDEA_RE.main()
   if(pm.LDA == 1):
      import iDEA_LDA2
   if(pm.MLP == 1):
      print('MLP: not yet implemented')
if(pm.NE==3):
   if(pm.MB == 1):
      iDEA_MB3.main()
   if(pm.REV == 1):
      import iDEA_RE
      iDEA_RE.main()
   if(pm.LDA == 1):
      import iDEA_LDA3
   if(pm.MLP == 1):
      print('MLP: not yet implemented')

# All jobs done
string = 'all jobs done \n'
sprint.sprint(string,2,0,pm.msglvl)
sprint.sprint(string,1,0,pm.msglvl)
 




