import pickle

def plot(Nx,Nt,timestep,file_name):
    input_file = open('ProbPsi(Nx=' + str(Nx) + ',Nt=' + str(Nt) + ').db','r')
    output_file = open(str(file_name) + '.dat','w')
    data = pickle.load(input_file)
    density = data[int(timestep)]
    for i in range(0,len(density)):
        output_file.write(str(i) + ' ' + str(density[i]) + '\n')      
    input_file.close()
    output_file.close()

if(__name__ == '__main__'):
    print('This program will extract a spacial plot of density for a given timestep from a ProbPsi.db file')
    Nx = int(input('(In ProbPsi file name) Nx = '))
    Nt = int(input('(In ProbPsi file name) Nt = ')) 
    t = int(input('Plot density at timestep = '))
    f = str(raw_input('name of output file = '))
    plot(Nx,Nt,t,f)
