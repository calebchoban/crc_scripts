import numpy as np
import utils


# This is a class that manages a galaxy/halo in a given
# snapshot, for either cosmological or isolated simulations.

class Halo:

    def __init__(self, sp, id=0):

        # the halo must be taken from a snapshot
        self.sp = sp
        self.k = -1 if sp.k==-1 else 0 # no snapshot, no halo
        self.id = id
        self.cosmological = sp.cosmological
        
        # these are linked to its parent snapshot
        self.header = sp.header
        self.gas = sp.gas
        self.DM = sp.DM
        self.disk = sp.disk
        self.bulge = sp.bulge
        self.star = sp.star
        self.BH = sp.BH
        self.part = sp.part

        return


    # load basic info to the class
    def load(self, mode='AHF', nclip=1000):

        if (self.k!=0): return # already loaded

        sp = self.sp

        # non-cosmological snapshots
        if (sp.cosmological==0):

            self.k = 1
            self.time = sp.time
            self.xc = sp.boxsize/2.
            self.yc = sp.boxsize/2.
            self.zc = sp.boxsize/2.
            self.rvir = sp.boxsize*2.
        
            return

        # cosmological, use AHF
        if (mode=='AHF'):
            
            AHF = sp.loadAHF()
    
            # catalog exists
            if (AHF.k==1):
                
                self.k = 1
                self.time = sp.time
                self.redshift = sp.redshift
                self.catalog = 'AHF'
                self.id = AHF.ID[self.id]
                self.host = AHF.hostHalo[self.id]
                self.npart = AHF.npart[self.id]
                self.ngas = AHF.n_gas[self.id]
                self.nstar = AHF.n_star[self.id]
                self.mvir = AHF.Mvir[self.id]
                self.mgas = AHF.M_gas[self.id]
                self.mstar = AHF.M_star[self.id]
                self.xc = AHF.Xc[self.id]
                self.yc = AHF.Yc[self.id]
                self.zc = AHF.Zc[self.id]
                self.rvir = AHF.Rvir[self.id]
                self.rmax = AHF.Rmax[self.id]
                self.vmax = AHF.Vmax[self.id]
                self.fhi = AHF.fMhires[self.id]

            # no catalog
            else:

                self.k = 0 # so you can re-load it
                self.redshift = sp.redshift
                self.xc = sp.boxsize/2.
                self.yc = sp.boxsize/2.
                self.zc = sp.boxsize/2.
                self.rvir = sp.boxsize*2.

        # cosmological, use rockstar
        else:
        
            rockstar = sp.loadrockstar(nclip=nclip)
        
            # catalog exists
            if (rockstar.k==1):
            
                self.k = 1
                self.time = sp.time
                self.redshift = sp.redshift
                self.catalog = 'rockstar'
                self.id = rockstar.id[self.id]
                self.rockstarid = rockstar.rockstarid[self.id]
                self.npart = rockstar.num_p[self.id]
                self.mvir = rockstar.mvir[self.id]
                self.xc = rockstar.x[self.id]
                self.yc = rockstar.y[self.id]
                self.zc = rockstar.z[self.id]
                self.rvir = rockstar.rvir[self.id]
                self.rmax = rockstar.rvmax[self.id]
                self.vmax = rockstar.vmax[self.id]
            
            # no catalog
            else:
        
                self.k = 0 # so you can re-load it
                self.redshift = sp.redshift
                self.xc = sp.boxsize/2.
                self.yc = sp.boxsize/2.
                self.zc = sp.boxsize/2.
                self.rvir = sp.boxsize*2.
    
        return


    # load particle for the snapshot
    def loadpart(self, ptype, header_only=0):

        part = self.header if header_only==1 else self.part[ptype]
        part.load()

        return part
    
    
    # load AHF particles for the halo
    def loadAHFpart(self):
    
        AHF = self.sp.loadAHF()
        PartID, PType = AHF.loadpart(id=self.id)
    
        return PartID, PType
    
    
    # this calls my own visual module
    def viewpart(self, ptype, field='None', method='simple', **kwargs):

        self.load()
        if (self.k==-1): return -1 # no halo

        part = self.part[ptype]; part.load()
        if (part.k==-1): return -1 # no valid particle

        # set center and boundaries
        if 'cen' not in kwargs: kwargs['cen'] = [self.xc,self.yc,self.zc]
        if 'L' not in kwargs: kwargs['L'] = self.rvir
        
        # add time label for the image
        kwargs['time'] = r"$z=%.1f$" % self.redshift if self.cosmological else r"$t=%.1f$" %self.time
        
        # check which field to show
        h, wt = None, part.m
        if (ptype==0):
            # for gas, check if we want a field from the list below
            if (field in ['nh','ne','z']) and (field in dir(part)): wt *= getattr(part,field)
        if (ptype==4):
            # for stars, check if we want luminosity from some band
            import colors
            if (field in colors.colors_available) and ('age' in dir(part)) and ('z' in dir(part)):
                wt *= colors.colors_table(part.age, part.z/0.02, band=field)
        if (method=='smooth'):
            h = part.h if ptype==0 else utils.get_particle_hsml(part.p[:,0],part.p[:,1],part.p[:,2])
        
        # now, call the routine
        import visual
        H = visual.make_projected_image(part.p, wt, h=h, method=method, **kwargs)
            
        return H
    

    # this calls PFH visualization module
    def image_maker(self, ptype, **kwargs):

        self.load()
        if (self.k==-1): return -1 # no halo
        
        # set center if needed
        if 'cen' not in kwargs: kwargs['cen'] = [self.xc, self.yc, self.zc]
        if 'L' not in kwargs: kwargs['L'] = self.rvir
        if 'fbase' not in kwargs: kwargs['fbase'] = "snap%03d_halo%d" %(self.sp.snum,self.id)

        import visual
        massmap = visual.image_maker(self.sp, ptype, **kwargs)
    
        return massmap


    # get star formation history using all stars in Rvir
    def get_SFH(self, dt=0.01, cum=0, cen=None, rout=1.0, kpc=0):

        self.load()
        if (self.k==-1): return 0., 0.

        try:
            xc, yc, zc = cen[0], cen[1], cen[2]
        except (TypeError,IndexError):
            xc, yc, zc = self.xc, self.yc, self.zc

        part = self.loadpart(4, header_only=0)
        if (part.k==-1): return 0., 0.

        p, sft, m = part.p, part.sft, part.m
        r = np.sqrt((p[:,0]-xc)**2+(p[:,1]-yc)**2+(p[:,2]-zc)**2)
        rmax = rout*self.rvir if kpc==0 else rout
        t, sfr = utils.SFH(sft[r<rmax], m[r<rmax], dt=dt, cum=cum, sp=self.sp)

        return t, sfr


    def calculate_zoom_center(self, ptype=4):

        # the halo does not exist
        if (self.k==-1): return 0., 0., 0.

        xc, yc, zc, rvir = self.xc, self.yc, self.zc, self.rvir
        part = self.loadpart(ptype, header_only=0)

        # particle does not exist
        if (part.k==-1): return xc, yc, zc

        # load particle coordinates
        p = part.p
        r = np.sqrt((p[:,0]-xc)**2+(p[:,1]-yc)**2+(p[:,2]-zc)**2)
        ok = r<rvir

        # there is no valid particle in rvir
        if (len(r[ok])==0): return xc, yc, zc

        # this is the place we want to start
        xc, yc, zc = np.median(p[ok,0]), np.median(p[ok,1]), np.median(p[ok,2])
        for rmax in [50,20,10,5,2,1]:
            if (rmax>rvir): continue
            # do 5 iteration at each level
            for i in range(5):
                r = np.sqrt((p[:,0]-xc)**2+(p[:,1]-yc)**2+(p[:,2]-zc)**2)
                ok = r<rmax
                xc, yc, zc = np.median(p[ok,0]), np.median(p[ok,1]), np.median(p[ok,2])

        return xc, yc, zc
