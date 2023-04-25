from . import snapshot


# this returns a Snapshot class
def load_snap(sdir, snum, cosmological=0, periodic_bound_fix=True):

    sp = snapshot.Snapshot(sdir, snum, cosmological=cosmological, periodic_bound_fix=periodic_bound_fix)

    return sp


# this returns a Particle class
def load_part(sdir, snum, ptype, cosmological=0, periodic_bound_fix=False):

    sp = load_snap(sdir, snum, cosmological=cosmological, periodic_bound_fix=periodic_bound_fix)
    part = sp.loadpart(ptype)

    return part 

# this returns a Header class
def load_header(sdir, snum, ptype, cosmological=0, periodic_bound_fix=False):

    sp = load_snap(sdir, snum, cosmological=cosmological, periodic_bound_fix=periodic_bound_fix)
    part = sp.loadheader(ptype)

    return part 



# this returns an AHF class
def load_AHF(sdir, snum, cosmological=0, hdir=None):

    sp = load_snap(sdir, snum, cosmological=cosmological)
    AHF = sp.loadAHF(hdir=hdir)

    return AHF


# this returns the primary halo/galaxy
def load_halo(sdir, snum, cosmological=0, id=-1, mode='AHF', hdir=None, periodic_bound_fix=False):

    sp = load_snap(sdir, snum, cosmological=cosmological, periodic_bound_fix=periodic_bound_fix)
    halo = sp.loadhalo(id=id, mode=mode, hdir=hdir)

    return halo


# this returns the primary galactic disk
def load_disk(sdir, snum, cosmological=0, id=-1, mode='AHF', hdir=None, periodic_bound_fix=False, rmax=20, height=5):

    sp = load_snap(sdir, snum, cosmological=cosmological, periodic_bound_fix=periodic_bound_fix)
    disk = sp.loaddisk(id=id, mode=mode, hdir=hdir, rmax=rmax, height=height)

    return disk