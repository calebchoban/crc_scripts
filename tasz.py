import numpy as np

def tasz(omeganot, littleh, interval=1000):
    amin = 1.0 / interval
    #amin = 0.001
    amax = 1.0 
    alist = []
    tlist = []
    for i in range (0, (interval+1)):
        a = amin + i*(amax-amin)/interval
        f = omeganot * a ** (-3.0) / (1.0 - omeganot)
        t = (2.0/3.0)* 9.779e9 * (1.0/littleh) * np.log((1.0 + (1.0+f)**0.5)/f**0.5) / (1.0-omeganot)**0.5
        print a,'  ',t/1e9
        alist.append(a)
        tlist.append(t/1e9)
    return alist, tlist

def tfora(a, omeganot, littleh):
    f = omeganot * a ** (-3.0) / (1.0 - omeganot)
    t = (2.0/3.0)* 9.779e9 * (1.0/littleh) * np.log((1.0 + (1.0+f)**0.5)/f**0.5) / (1.0-omeganot)**0.5
    return (t/1e9)
    
def hubble_param(a, omeganot, littleh):
	omega_lamda = 1.0 - omeganot
	z = 1.0 / a - 1.0
	H = np.sqrt(omeganot * (1.0 + z)**3.0 + omega_lamda) * 100.0 * littleh
	return H
