import os
import h5py
import numpy as np

def check_snap_exist(sdir, snum):
    """
    Check if the snapshot with the given number exists in the given directory and returns the number of files
    it comprises if it exists.

    Parameters
    ----------
    sdir : string
        Directory of snapshot
    snum : int
        Number of snapshot

    Returns
    -------
    nsnap : int
        Number of files that make up snapshot or 0 if no snapshot exists

    """
    
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


def get_snap_file_name(sdir, snum, nsnap, i):
    """
    Get name of file for snapshot. If snapshot is multiple files it will return the name of the specified subfile.
    Parameters
    ----------
    sdir : string
        Directory of snapshot
    snum : int
        Number of snapshot
    nsnap : int
        Number of subfiles making up snapshot
    i : int
        Number of subfile

    Returns
    -------
    snapfile : string
        Full name of snapshot in specified directory.
        
    """
    if (nsnap==1):
        snapfile = sdir + "/snapshot_%03d.hdf5" %snum
    else:
        snapfile = sdir + "/snapdir_%03d/snapshot_%03d.%d.hdf5" %(snum,snum,i)

    return snapfile
