"""These are the parameters for the LDAs developed in [Entwistle2018]_ from finite slab systems 
and the HEG.
"""


###############
# Finite LDAs #
###############

# parameters for \varepsilon_{xc}(n)
exc_lda = {} 
exc_lda[1] = {
   'a' : -1.2202,
   'b' : 3.6838,
   'c' : -11.254,
   'd' : 23.169,
   'e' : -26.299,
   'f' : 12.282,
   'g' : 0.74876,
}
exc_lda[2] =  {
   'a' : -1.0831,
   'b' : 2.7609,
   'c' : -7.1577,
   'd' : 12.713,
   'e' : -12.755,
   'f' : 5.3817,
   'g' : 0.70955,
}
exc_lda[3] =  {
   'a' : -1.1002,
   'b' : 2.9750,
   'c' : -8.1618,
   'd' : 15.169,
   'e' : -15.776,
   'f' : 6.8494,
   'g' : 0.70907,
}

# parameters for V_{xc}(n)
vxc_lda = {}  
for n in [1,2,3]:
    eps = exc_lda[n]
    a = eps['a']
    b = eps['b']
    c = eps['c']
    d = eps['d']
    e = eps['e']
    f = eps['f']
    g = eps['g']

    vxc_lda[n] = {
      'a' : (g+1)*a,
      'b' : (g+2)*b,
      'c' : (g+3)*c,
      'd' : (g+4)*d,
      'e' : (g+5)*e,
      'f' : (g+6)*f,
      'g' : g,
    }

# parameters for dV_{xc}(n)/dn
dlda = {} 
for n in [1,2,3]:
    eps = exc_lda[n]
    a = eps['a']
    b = eps['b']
    c = eps['c']
    d = eps['d']
    e = eps['e']
    f = eps['f']
    g = eps['g']

    dlda[n] = {
      'a' : a*g*(g+1.0),
      'b' : b*(g**2+3.0*g+2.0),
      'c' : c*(g**2+5.0*g+6.0),
      'd' : d*(g**2+7.0*g+12.0),
      'e' : e*(g**2+9.0*g+20.0),
      'f' : f*(g**2+11.0*g+30.0),
      'g' : g-1.0,
    }

###########
# HEG LDA #
###########

# parameters for \varepsilon_{x}(n)
ex_lda = {} 
ex_lda['heg'] = {
   'a' : -1.1511,
   'b' : 3.3440,
   'c' : -9.7079,
   'd' : 19.088,
   'e' : -20.896,
   'f' : 9.4861,
   'g' : 0.73586,
}

# parameters for \varepsilon_{c}(n)
ec_lda = {} 
ec_lda['heg'] = {
   'a' :  0.0009415195,
   'b' :  0.2601,
   'c' :  0.06404,
   'd' :  0.000248,
   'e' :  0.00000261,
   'f' :  1.254,
   'g' :  28.8, 
}

# parameters for V_{x}(n)
vx_lda = {} 
eps = ex_lda['heg']
a = eps['a']
b = eps['b']
c = eps['c']
d = eps['d']
e = eps['e']
f = eps['f']
g = eps['g']

vx_lda['heg'] = {
   'a' : (g+1)*a,
   'b' : (g+2)*b,
   'c' : (g+3)*c,
   'd' : (g+4)*d,
   'e' : (g+5)*e,
   'f' : (g+6)*f,
   'g' : g,
}
