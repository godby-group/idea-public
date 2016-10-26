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
import splash
import sprint
import SPiDEA
import EXT2
import EXT3
import NON
import LDA
import MLP
import MBPT
import LAN
import parameters as pm

# Draw splash to screen
splash.draw(pm.run.msglvl)
print('run name: ' + str(pm.run.name))

# Execute required jobs
if(pm.sys.NE == 1):
   if(pm.run.EXT == True):
      SPiDEA.main()
   if(pm.ext.RE == True):
      import RE
      RE.main('ext')
if(pm.sys.NE == 2):
   if(pm.run.EXT == True):
      EXT2.main()
   if(pm.ext.RE == True):
      import RE
      RE.main('ext')
if(pm.sys.NE == 3):
   if(pm.run.EXT == True):
      EXT3.main()
   if(pm.ext.RE == True):
      import RE
      RE.main('ext')
if(pm.sys.NE >= 4):
   if(pm.run.EXT == True):
      print('EXT: cannot run exact with more than 3 electrons')
if(pm.run.NON == True):
      NON.main()
if(pm.non.RE == True):
      import RE
      RE.main('non')
if(pm.run.LDA == True):
      LDA.main()
if(pm.run.MLP == True):
      MLP.main()
if(pm.run.HF == True):
      import HF
if(pm.hf.RE == True):
      import RE
      RE.main('hf')
if(pm.run.MBPT == True):
      MBPT.main()
if(pm.mbpt.RE == True):
      import RE
      RE.main('mbpt')
if(pm.run.LAN == True):
      LAN.main()

# All jobs done
string = 'all jobs done \n'
sprint.sprint(string,2,0,pm.run.msglvl)
sprint.sprint(string,1,0,pm.run.msglvl)
