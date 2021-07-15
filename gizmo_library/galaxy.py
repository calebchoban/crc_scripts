import numpy as np
from . import utils


# This is a class that manages a galaxy/halo in a given
# snapshot, for either cosmological or isolated simulations.

class Halo(object):

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

        # Set later if you want to zoom in
        self.zoom = False

        return


    # Set zoom-in region if you don't want all data in rvir
    def set_zoom(self, rout=1.0, kpc=False, ptype=4):

        # How far out do you want to load data, default is rvir
        self.rout = rout
        self.outkpc = kpc
        self.zoom_ptype = ptype
        self.zoom = True

        return


    # load basic info to the class
    def load(self, mode='AHF'):

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
            self.Lhat = np.array([0,0,1.])

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
                self.Lhat = AHF.Lhat[self.id]

            # no catalog
            else:

                self.k = 0 # so you can re-load it
                self.redshift = sp.redshift
                self.xc = sp.boxsize/2.
                self.yc = sp.boxsize/2.
                self.zc = sp.boxsize/2.
                self.rvir = sp.boxsize*2.
                self.Lhat = np.array([0,0,1.])
    
        return


    # Returns the maximum radius
    def get_rmax(self):
        return self.rvir


    # load all particles in the halo/galaxy centered on halo center
    def loadpart(self, ptype):

        part = self.part[ptype]

        part.load()
        if self.zoom:
            rmax = self.rout*self.rvir if not self.outkpc else self.rout
            xc,yc,zc = self.calculate_zoom_center(ptype=self.zoom_ptype)
        else:
            rmax = self.rvir
            xc=self.xc; yc=self.yc; zc=self.zc
        part.center([xc,yc,zc])
        in_halo = np.sum(np.power(part.p,2),axis=1) <= np.power(rmax,2.)
        part.mask(in_halo)

        return part


    # load particles in the zoom in center of halo/galaxy out to rout*rvir or rout in kpc
    def load_zoompart(self, ptype, zoom_ptype=4, rout=0.2, kpc=0):

        xc,yc,zc = self.calculate_zoom_center(ptype=zoom_ptype)
        part = self.part[ptype]
        part.load()
        part.center([xc,yc,zc])
        rmax = rout*self.rvir if kpc==0 else rout
        in_zoom = np.sum(np.power(part.p,2),axis=1) <= np.power(rmax,2.)
        part.mask(in_zoom)

        return part


    def loadheader(self):

        header=self.header
        header.load()

        return header

    
    # load AHF particles for the halo
    def loadAHFpart(self):
    
        AHF = self.sp.loadAHF()
        PartID, PType = AHF.loadpart(id=self.id)
    
        return PartID, PType
    

    # centers the particles for the given viewing angle
    def centerpart(self, ptype, view='faceon'):
        Lz_hat = np.array([0.,0.,1.])
        Ly_hat = np.array([0.,1.,0.])
        Lhat = self.Lhat
        if view == 'faceon':
            rot_matrix = utils.calc_rotate_matrix(Lz_hat,Lhat)
            self.sp.rotate_coords(rot_matrix)
        elif view == 'edgeon':
            rot_matrix = utils.calc_rotate_matrix(Lz_hat,Lhat)
            self.sp.rotate_coords(rot_matrix)
            rot_matrix = utils.calc_rotate_matrix(Lz_hat,Ly_hat)
            self.sp.rotate_coords(rot_matrix)
        elif view == 'random':
            L_rand = np.random.rand(3)
            L_rand = L_rand / np.linalg.norm(L_rand)
            rot_matrix = utils.calc_rotate_matrix(L_rand,Lhat)
            self.sp.rotate_coords(rot_matrix)

        self.Lhat = Lz_hat


        return


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

        part = self.loadpart(4)
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
        part = self.sp.loadpart(ptype)

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


class Disk(Halo):

    def __init__(self, sp, id=0, rmax=20, height=5):
        super(Disk, self).__init__(sp, id=id)
        # specify the dimensions of the disk
        self.rmax = rmax
        self.height = height

        return


 # load basic info to the class
    def load(self, mode='AHF'):

        if (self.k!=0): return # already loaded

        sp = self.sp
        
        # non-cosmological snapshots
        if (sp.cosmological==0):

            self.k = 1
            self.time = sp.time
            # Get center from average of gas particles
            part = self.sp.loadpart(0)
            self.xc = np.average(part.p[:,0],weights=part.m)
            self.yc = np.average(part.p[:,1],weights=part.m)
            self.zc = np.average(part.p[:,2],weights=part.m)
            self.rvir = sp.boxsize*2.
            self.Lhat = np.array([0,0,1.])
        
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
                self.Lhat = AHF.Lhat[self.id]

            # no catalog
            else:

                self.k = 0 # so you can re-load it
                self.redshift = sp.redshift
                # Get center from average of gas particles
                part = self.sp.loadpart(0)
                self.xc = np.average(part.p[:,0],weights=part.m)
                self.yc = np.average(part.p[:,1],weights=part.m)
                self.zc = np.average(part.p[:,2],weights=part.m)
                self.rvir = sp.boxsize*2.
                self.Lhat = np.array([0,0,1.])
        
                self.xc = sp.boxsize/2.
                self.yc = sp.boxsize/2.
                self.zc = sp.boxsize/2.
                self.rvir = sp.boxsize*2.
                self.Lhat = np.array([0,0,1.])
    
        return


    # load all particles in the galactic disk
    def loadpart(self, ptype):

        part = self.part[ptype]

        part.load()
        part.center([self.xc,self.yc,self.zc])
        zmag = np.dot(part.p,self.Lhat)
        r_z = np.zeros(np.shape(part.p))
        r_z[:,0] = zmag*self.Lhat[0]
        r_z[:,1] = zmag*self.Lhat[1]
        r_z[:,2] = zmag*self.Lhat[2]
        r_s = np.subtract(part.p,r_z)
        smag = np.sqrt(np.sum(np.power(r_s,2),axis=1))
        in_disk = np.logical_and(np.abs(zmag) <= self.height, smag <= self.rmax)
        part.mask(in_disk)

        return part

    # Returns the maximum radius
    def get_rmax(self):
        return self.rmax
