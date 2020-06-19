import numpy as np
import gizmo_lib.snapshot
import constant
import xma_cosmology


# this returns a Snapshot class
def loadsnap(sdir, snum, cosmological=0):

    sp = gizmo_lib.snapshot.Snapshot(sdir, snum, cosmological=cosmological)

    return sp


# this returns a Herder/Particle class
def loadpart(sdir, snum, ptype, cosmological=0, header_only=0):

    sp = loadsnap(sdir, snum, cosmological=cosmological)
    part = sp.loadpart(ptype, header_only=header_only)

    return part 


# this returns an AHF class
def loadAHF(sdir, snum, cosmological=0, hdir=None):

    sp = loadsnap(sdir, snum, cosmological=cosmological)
    AHF = sp.loadAHF(hdir=hdir)

    return AHF


# this returns a rockstar class
def loadrockstar(sdir, snum, cosmological=0, nclip=1000):

    sp = loadsnap(sdir, snum, cosmological=cosmological)
    rockstar = sp.loadrockstar(nclip=nclip)

    return rockstar


# this returns the primary halo/galaxy
def loadhalo(sdir, snum, cosmological=0, id=-1, mode='AHF', hdir=None):

    sp = loadsnap(sdir, snum, cosmological=cosmological)
    halo = sp.loadhalo(id=id, mode=mode, hdir=hdir)

    return halo


def PhysicalConstant():

    return constant.PhysicalConstant()


def Cosmology(h=0.68, omega=0.31):

    return cosmology.Cosmology(h=h, omega=omega)
