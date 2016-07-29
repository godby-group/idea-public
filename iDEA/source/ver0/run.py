######################################################################################
# Name: Run                                                                          #
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
import iDEA_EXT2
import iDEA_EXT3
import iDEA_NON
import iDEA_LDA
import iDEA_MLP
import iDEA_MBPT
import iDEA_LAN
import parameters as pm

# Draw splash to screen
splash.draw(pm.msglvl)
print('run name: ' + str(pm.run_name))

# Execute required jobs
if(pm.NE==1):
   if(pm.EXT == 1):
      SPiDEA.main()
   if(pm.EXT_RE == 1):
      import iDEA_RE
      iDEA_RE.main('ext')
if(pm.NE==2):
   if(pm.EXT == 1):
      iDEA_EXT2.main()
   if(pm.EXT_RE == 1):
      import iDEA_RE
      iDEA_RE.main('ext')
if(pm.NE==3):
   if(pm.EXT == 1):
      iDEA_EXT3.main()
   if(pm.EXT_RE == 1):
      import iDEA_RE
      iDEA_RE.main('ext')
if(pm.NE >= 4):
   if(pm.EXT == 1):
      print('EXT: cannot run exact with more than 3 electrons')
if(pm.NON == 1):
      iDEA_NON.main()
if(pm.NON_RE == 1):
      import iDEA_RE
      iDEA_RE.main('non')
if(pm.LDA == 1):
      iDEA_LDA.main()
if(pm.MLP == 1):
      iDEA_MLP.main()
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
if(pm.LAN == 1):
      iDEA_LAN.main()

# All jobs done
string = 'all jobs done \n'
sprint.sprint(string,2,0,pm.msglvl)
sprint.sprint(string,1,0,pm.msglvl)
