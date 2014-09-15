import pickle
import parameters as pm

def plot(timestep):
    input_file = open('CDensity/ProbPsi(Nx=' + str(pm.jmax) + ',Nt=' + str(pm.imax) + ').db','r')
    output_file = open('CDensity.dat','w')
    data = pickle.load(input_file)
    density = data[int(timestep)]
    for i in range(0,len(density)):
        output_file.write(str(i) + ' ' + str(density[i]) + '\n')      
    input_file.close()
    output_file.close()

if(__name__ == '__main__'):
    x = int(input('Plot density at timestep = '))
    plot(x)
