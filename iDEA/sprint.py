import sys

def sprint(text, n, s, msglvl):
    if(n == msglvl):
	if(s == 1):
	    sys.stdout.write('\033[K')
	    sys.stdout.flush()
	    sys.stdout.write('\r' + text)
	    sys.stdout.flush()
        else:
            print(text)

