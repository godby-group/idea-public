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
import iDEA_NON
import iDEA_MBPT
import parameters as pm

# Draw splash to screen
splash.draw(pm.msglvl)
print('run name: ' + str(pm.run_name))

# Execute required jobs
if(pm.NE==1):
   if(pm.MB == 1):
      SPiDEA.main()
   if(pm.MB_RE == 1):
      import iDEA_RE
      iDEA_RE.main('ext')
if(pm.NE==2):
   if(pm.MB == 1):
      iDEA_MB2.main()
   if(pm.MB_RE == 1):
      import iDEA_RE
      iDEA_RE.main('ext')
if(pm.NE==3):
   if(pm.MB == 1):
      iDEA_MB3.main()
   if(pm.MB_RE == 1):
      import iDEA_RE
      iDEA_RE.main('ext')
if(pm.NE >= 4):
   if(pm.MB == 1):
      print('many body: cannot run many body with more than 3 electrons')
if(pm.NON == 1):
      iDEA_NON.main()
if(pm.NON_RE == 1):
      import iDEA_RE
      iDEA_RE.main('non')
if(pm.LDA == 1):
      import iDEA_LDA
if(pm.MLP == 1):
      import iDEA_MLP
if(pm.HF == 1):
      import iDEA_HF
if(pm.HF_RE == 1):
      import iDEA_RE
      iDEA_RE.main('hf')
if(pm.MBPT == 1):
      iDEA_MBPT.main()
if(pm.MBPT_RE == 1):
      import iDEA_RE
      iDEA_RE.main('mbpt')

# All jobs done
string = 'all jobs done \n'
sprint.sprint(string,2,0,pm.msglvl)
sprint.sprint(string,1,0,pm.msglvl)
