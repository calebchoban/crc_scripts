import numpy as np
import constant


# cosmological calculator 
# Note:
#    This is a FLAT universe.
#    This is in MATTER-dominated era.

pc = constant.PhysicalConstant()

class Cosmology:

    def __init__(self, h=0.68, omega=0.31):

        self.h = h
        self.OmegaM = omega
        self.OmegaL = 1 - omega
        self.rho_crit = 3*(100*self.h*pc.km/pc.Mpc)**2 / (8*pc.pi*pc.G)
        self.DH = pc.c/(100*h*pc.km) # in Mpc

        self.comoving_distance = np.vectorize(self.Dc)
        self.angular_distance = np.vectorize(self.DA)
        self.luminosity_distance = np.vectorize(self.DL)

        return


    def cosmic_time(self, z):

        h, OmegaM = self.h, self.OmegaM

        x = OmegaM / (1.0-OmegaM) * (1+z)**3
        t = (2.0/(3.0*np.sqrt(1.0-OmegaM))) * np.log(np.sqrt(x)/(-1.0+np.sqrt(1.+x)))
        t *= (13.777*(0.71/h)) # in Gyr

        return t
    
    
    def cosmic_time_to_redshift(self, t):
    
        zgrid = 10**np.linspace(2, 0, 1000)
        tgrid = cosmic_time(zgrid)
    
        return np.interp(t, tgrid, zgrid)


    def Ez(self, z):

        return np.sqrt(self.OmegaL+self.OmegaM*(1+z)**3)


    # comoving distance
    def Dc(self, z):

        a = 1.0/(1.0+z)
        da = min(0.001, a/10)
        n = int((1.0-a)/da) + 10

        agrid = np.linspace(1, a, n)
        zgrid = 1.0/agrid - 1.0
        dgrid = 1.0/self.Ez(zgrid)

        Dc = np.trapz(dgrid, zgrid)
        Dc *= self.DH # in cMpc

        return Dc


    # angular diameter distance
    def DA(self, z):

        return self.Dc(z)/(1.0+z)


    # luminosity distance
    def DL(self, z):

        return self.Dc(z)*(1.0+z)


    # comoving volume
    def comoving_volume(self, z1, z2):

        a1 = 1.0/(1.0+z1)
        a2 = 1.0/(1.0+z2)
        da = min(0.001, a1/10, a2/10)
        n = int(np.abs(a1-a2)/da) + 10

        agrid = np.linspace(a1, a2, n)
        zgrid = 1.0/agrid - 1.0
        vgrid = self.comoving_distance(zgrid)**2/self.Ez(zgrid)

        Vc = np.trapz(vgrid, zgrid)
        Vc *= self.DH # cMpc^3 per solid angle
        Vc *= pc.sqamin # cMpc^3 per square arcmin

        return Vc
    
    
    # surface brightness
    def surface_brightness(self, input, z, ismag=True):
        
        if (ismag is True):
            
            # from mag arcsec^-2 to erg s^-1 Hz^-1 kpc^-2
            dx = 1.0e3*self.DA(z)*pc.arcsec # 1 arcsec in kpc
            Lnu = 10**(-(input+48.6)/2.5)*4*np.pi*(pc.Mpc*self.DL(z))**2/(1.0+z) # in erg/s/Hz
            output = Lnu / dx**2
        
        else:
        
            # from erg s^-1 Hz^-1 kpc^-2 to mag arcsec^-2
            dx = 1.0e3*self.DA(z)*pc.arcsec # 1 arcsec in kpc
            mag = -2.5*np.log10((1.0+z)*input*dx**2/(4*np.pi*(pc.Mpc*self.DL(z))**2))-48.6
            output = mag

        return output
