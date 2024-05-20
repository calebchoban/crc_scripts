import os
import h5py
import numpy as np

# check if snap exists and return number of files
def check_snap_exist(sdir, snum):
    
    # single file case
    snapfile = sdir + "/snapshot_%03d.hdf5" %snum
    if (os.path.isfile(snapfile)): return 1
    
    # multiple files
    snapfile = sdir + "/snapdir_%03d/snapshot_%03d.0.hdf5" %(snum,snum)
    if (os.path.isfile(snapfile)):
        f = h5py.File(snapfile, 'r')
        nsnap = f['Header'].attrs['NumFilesPerSnapshot']
    else:
        print("Snapshot",sdir + "/snapshot_%03d.hdf5" %snum,"doesn't exist.")
        return 0
    
    for i in np.arange(1,nsnap,1):
        snapfile = sdir + "/snapdir_%03d/snapshot_%03d.%d.hdf5" %(snum,snum,i)
        if (not os.path.isfile(snapfile)):
            print("Snapshot is not complete.")
            return 0
    
    return nsnap


# this function returns snapshot file name
def get_snap_file_name(sdir, snum, nsnap, i):
    if (nsnap==1):
        snapfile = sdir + "/snapshot_%03d.hdf5" %snum
    else:
        snapfile = sdir + "/snapdir_%03d/snapshot_%03d.%d.hdf5" %(snum,snum,i)

    return snapfile
