import numpy as np
from scipy.integrate import quad
from .. import config


def MRN_dnda(a):
    return np.power(a,-3.5)

def MRN_dmda(a):
    return 4/3*np.pi * 1 * np.power(a,3)*np.power(a,-3.5)

def MRN_dmda_update(a,da):
    return 4/3*np.pi * 1 * np.power(a+da,3)*np.power(a,-3.5)

# Determines the change in grain distribution given a constant change in grain size
def change_in_grain_distribution(da,amin=1E-9,amax=5E-6,bin_num=1000):
    # Assume MRN distribution 
    a = np.logspace(np.log10(amin),np.log10(amax),bin_num+1)
    # Append bins beyond min and max
    a=np.append(a,np.inf); a=np.append(-np.inf,a)
    bin_num+=2
    N_bin = np.zeros(bin_num); M_bin = np.zeros(bin_num)
    N_update = np.zeros(bin_num); M_update = np.zeros(bin_num)
    for i in range(bin_num):
        # Nothing outside of the min/max size range
        if i==0 or i == bin_num-1:
            N_bin[i]=0
            M_bin[i]=0
        else:
            N_bin[i] = quad(MRN_dnda,a[i],a[i+1])[0]
            M_bin[i] = quad(MRN_dmda,a[i],a[i+1])[0]

    for j in range(bin_num):
        aj_upper = a[j+1]; aj_lower = a[j];
        #print('j',j,aj_lower,aj_upper)
        for i in range(bin_num):
            ai_upper = a[i+1]; ai_lower = a[i];
            #print('i',i,ai_lower,ai_upper)
            intersect = [np.max([aj_lower-da, ai_lower]), np.min([aj_upper-da, ai_upper])]
            if intersect[0]>intersect[1] or intersect[0]>amax or intersect[1]<amin: 
                #print('no intersect')
                continue
            else:
               # Nothing beyond min/max grain size
               #print('int before',intersect)
               intersect[0]=np.max([intersect[0],amin])
               intersect[1]=np.min([intersect[1],amax])
               #print('int',intersect)
               N_update[j] += quad(MRN_dnda,intersect[0],intersect[1])[0]
               M_update[j] += quad(MRN_dmda_update,intersect[0],intersect[1],args=(da))[0]

    #print(N_bin,N_update)
    print(np.sum(N_bin),np.sum(N_update))
    #print(M_bin,M_update)
    print(np.sum(M_bin),np.sum(M_update))
    print('Change in number:',np.sum(N_update[1:-1])/np.sum(N_bin))
    print('Change in mass:',np.sum(M_update[1:-1])/np.sum(M_bin))