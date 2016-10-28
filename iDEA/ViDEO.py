######################################################################################
# Name: ViDEO (Visualise iDEA Outputs)                                               #
######################################################################################
# Author(s): Jack Wetherell                                                          #
######################################################################################
# Description:                                                                       #
# Visualise iDEA outputs to .dat .png .avi from .db                                  #
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
import sys
import pickle
import parameters as pm

# Print splash
print('                                                                ')
print('              *   *   *    ****     *****     ****              ')
print('              *   *        *   *    *        *    *             ')
print('              *   *   *    *    *   *       *      *            ')
print('              *   *   *    *     *  *****   *      *            ')
print('              *   *   *    *    *   *       *      *            ')
print('               * *    *    *   *    *        *    *             ')
print('                *     *    ****     *****     ****              ')
print('                                                                ')
print('  +------------------------------------------------------------+')
print('  |                    Visualise iDEA Outputs                  |')
print('  |                                                            |')
print('  |                   Created by Jack Wetherell                |')
print('  |                    The University of York                  |')
print('  +------------------------------------------------------------+')
print('                                                                ')

# Function to save ground-state data to a dat file
def save_data_gs(filename,dx,L):
   input_file = open('raw/' + str(filename) + '.db', 'r')
   output_file = open('data/' + str(filename) + '.dat', 'w')
   data = pickle.load(input_file)
   for i in range(0,len(data)):
      x = -0.5*L + dx*float(i)
      output_file.write(str(x) + ' ' + str(data[i]) + '\n')
   input_file.close()
   output_file.close()

# Function to save ground-state data to a png image
def save_plot_gs(filename,dx,L):
   input_file = open('raw/' + str(filename) + '.db', 'r')
   script_file = open('temp_script','w')
   output_file = open(str(filename) + '.dat', 'w')
   data = pickle.load(input_file)
   for i in range(0,len(data)):
      x = -0.5*L + dx*float(i)
      output_file.write(str(x) + ' ' + str(data[i]) + '\n')
   input_file.close()
   output_file.close()
   script_file.write('set terminal png size 1920,1080\n')
   script_file.write('set output "plots/' + str(filename) + '.png"\n')
   script_file.write("plot '" + str(filename) + ".dat'" + " w lines\n")
   script_file.close()
   os.system('gnuplot temp_script')
   os.system('rm temp_script')
   os.system('rm *.dat')

# Function to save timestep of time dependant data to a dat file
def save_data_td(filename,dx,L,save_data_timestep):
   input_file = open('raw/' + str(filename) + '.db', 'r')
   output_file = open('data/' + str(filename) + '_' + str(save_data_timestep) + '.dat', 'w')
   data = pickle.load(input_file)[save_data_timestep]
   for i in range(0,len(data)):
      x = -0.5*L + dx*float(i)
      output_file.write(str(x) + ' ' + str(data[i]) + '\n')
   input_file.close()
   output_file.close()

# Function to save timestep of time dependant data to a png image
def save_plot_td(filename,dx,L,save_plot_timestep):
   input_file = open('raw/' + str(filename) + '.db', 'r')
   script_file = open('temp_script','w')
   output_file = open(str(filename) + '_' + str(save_plot_timestep) + '.dat', 'w')
   data = pickle.load(input_file)[save_plot_timestep]
   for i in range(0,len(data)):
      x = -0.5*L + dx*float(i)
      output_file.write(str(x) + ' ' + str(data[i]) + '\n')
   input_file.close()
   output_file.close()
   script_file.write('set terminal png size 1920,1080\n')
   script_file.write('set output "plots/' + str(filename) + '_' + str(save_plot_timestep) + '.png"\n')
   script_file.write("plot '" + str(filename) + "_" + str(save_plot_timestep) + ".dat'" + " w lines\n")
   script_file.close()
   os.system('gnuplot temp_script')
   os.system('rm temp_script')
   os.system('rm *.dat')

# Function to animate time dependent pickle file
def animate_mp4(filename,dx,L,step):
   input_file = open('raw/' + str(filename) + '.db', 'r')
   raw_data = pickle.load(input_file)
   for i in range(0,len(raw_data),step):
       spring = 'animating: ' + str(int(100.0*float(float(i)/float(len(raw_data))))) + '%'
       sprint(spring,1,1,1)
       file_name = '_tmp%07d.dat'%i
       output_file = open(file_name, 'w')
       script_file = open('temp_script','w')
       data = raw_data[i]
       for j in range(0,len(data)):
           x = -0.5*L + dx*float(j)
           output_file.write(str(x) + ' ' + str(data[j]) + '\n')
       script_file.write('set terminal png size 1920,1080 \n')
       frame_name = '_tmp%07d.png'%i
       script_file.write('set output "' + str(frame_name) + '"\n')
       script_file.write("plot '" + str(file_name) + "'" + " w lines\n")
       output_file.close()
       script_file.close()
       os.system('gnuplot temp_script')
       os.system('rm temp_script')
       os.system('rm *.dat')
   print
   os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o " + 'animations/' + str(filename) + ".avi")
   os.system('rm *.png')
   input_file.close()

# Function to print to screen replacing last line
def sprint(text, n, s, msglvl):
    if(n == msglvl):
	if(s == 1):
	    sys.stdout.write('\033[K')
	    sys.stdout.flush()
	    sys.stdout.write('\r' + text)
	    sys.stdout.flush()
        else:
            print(text)

# Gather file information from user
run_name = pm.run.name
NE = int(input('number of electrons: '))
td = bool(input('is the data ground state or time dependant (gs=0,td=1): '))
approx = int(input('enter which approximation to use (exact=0,NON=1,LDA=2,MLP=3,HF=4,MBPT=5,LAN=6): '))
data = int(input('enter which quantity to plot (DEN=0,CUR=1,VXT=2,VKS=3,VH=4,VXC=5,ELF=6): '))
N = pm.sys.grid
L = pm.sys.xmax*2
dx = float(L/(N-1))
filename = str(run_name) + '_' + str(NE)
if(td):
   filename = filename + 'td_'
else:
   filename = filename + 'gs_' 
if(approx == 0):
   filename = filename + 'ext_'
if(approx == 1):
   filename = filename + 'non_'
if(approx == 2):
   filename = filename + 'lda_'
if(approx == 3):
   filename = filename + 'mlp_'
if(approx == 4):
   filename = filename + 'hf_'
if(approx == 5):
   filename = filename + 'mbpt_'
if(approx == 6):
   filename = filename + 'lan_'
if(data == 0):
   filename = filename + 'den'
if(data == 1):
   filename = filename + 'cur'
if(data == 2):
   filename = filename + 'vxt'
if(data == 3):
   filename = filename + 'vks'
if(data == 4):
   filename = filename + 'vh'
if(data == 5):
   filename = filename + 'vxc'
if(data == 6):
   filename = filename + 'elf'

# Determine what the user wants to be processed
if(td==0):
   save_data = bool(input('save to data file (0=no,1=yes): '))
   save_plot = bool(input('save to png image (0=no,1=yes): '))
if(td==1):
   save_data = bool(input('save to data file (0=no,1=yes): '))
   if(save_data):
       save_data_timestep = int(input('timestep to save data: '))
   save_plot = bool(input('save to png image (0=no,1=yes): '))
   if(save_plot):
       save_plot_timestep = int(input('timestep to plot image: '))
   animate = bool(input('save to avi video (0=no,1=yes): '))
   if(animate):
      step = int(input('skip every n frames (1=animate all frames, 2=skip every 2 frames, etc): n = '))

# Process data in specified way
if(td==0):
   if(save_data):
      save_data_gs(filename,dx,L)
   if(save_plot):
      save_plot_gs(filename,dx,L)
if(td==1):
   if(save_data):
      save_data_td(filename,dx,L,save_data_timestep)
   if(save_plot):
      save_plot_td(filename,dx,L,save_plot_timestep)
   if(animate):
      animate_mp4(filename,dx,L,step)

# Clear generated files
os.system('rm *.pyc')
