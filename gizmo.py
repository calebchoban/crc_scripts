import numpy as np
import gizmo_library.snapshot
import constant
import xma_cosmology


# this returns a Snapshot class
def loadsnap(sdir, snum, cosmological=0, periodic_bound_fix=True):

    sp = gizmo_lib.snapshot.Snapshot(sdir, snum, cosmological=cosmological, periodic_bound_fix=periodic_bound_fix)

    return sp


# this returns a Header/Particle class
def loadpart(sdir, snum, ptype, cosmological=0, header_only=0):

    sp = loadsnap(sdir, snum, cosmological=cosmological)
    part = sp.loadpart(ptype, header_only=header_only)

    return part 


# this returns an AHF class
def loadAHF(sdir, snum, cosmological=0, hdir=None):

    sp = loadsnap(sdir, snum, cosmological=cosmological)
    AHF = sp.loadAHF(hdir=hdir)

    return AHF


# this returns the primary halo/galaxy
def loadhalo(sdir, snum, cosmological=0, id=-1, mode='AHF', hdir=None):

    sp = loadsnap(sdir, snum, cosmological=cosmological)
    halo = sp.loadhalo(id=id, mode=mode, hdir=hdir)

    return halo