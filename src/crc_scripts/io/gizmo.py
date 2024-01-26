from . import snapshot


# this returns a Snapshot class
def load_snap(sdir, snum, cosmological=1):

    sp = snapshot.Snapshot(sdir, snum, cosmological=cosmological)

    return sp


# this returns a Particle class
def load_part(sdir, snum, ptype, cosmological=1):

    sp = load_snap(sdir, snum, cosmological=cosmological)
    part = sp.loadpart(ptype)

    return part 

# this returns a Header class
def load_header(sdir, snum, ptype, cosmological=1):

    sp = load_snap(sdir, snum, cosmological=cosmological)
    part = sp.loadheader(ptype)

    return part 



# this returns an AHF class
def load_AHF(sdir, snum, cosmological=1, hdir=None):

    sp = load_snap(sdir, snum, cosmological=cosmological)
    AHF = sp.loadAHF(hdir=hdir)

    return AHF


# this returns the primary halo/galaxy
def load_halo(sdir, snum, cosmological=1, id=-1, mode='AHF', hdir=None):

    sp = load_snap(sdir, snum, cosmological=cosmological)
    halo = sp.loadhalo(id=id, mode=mode, hdir=hdir)

    return halo


# this returns the primary galactic disk
def load_disk(sdir, snum, cosmological=1, id=-1, mode='AHF', hdir=None, rmax=20, height=5):

    sp = load_snap(sdir, snum, cosmological=cosmological)
    disk = sp.loaddisk(id=id, mode=mode, hdir=hdir, rmax=rmax, height=height)

    return disk