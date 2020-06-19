import numpy as np


class PhysicalConstant:

	def __init__(self):

		# distances
		self.km = 1.0e5
		self.AU = 1.496e13
		self.pc = 3.085678e18
		self.kpc = 1.0e3*self.pc
		self.Mpc = 1.0e6*self.pc
		self.Gpc = 1.0e9*self.pc

		# time
		self.yr = 3.1536e7
		self.kyr = 1.0e3*self.yr
		self.Myr = 1.0e6*self.yr
		self.Gyr = 1.0e9*self.yr

		# mass
		self.mH = 1.6737e-24
		self.Msun = 1.989e33

		# physical constant
		self.eV = 1.60218e-12
		self.kB = 1.38065e-16
		self.c = 2.99792458e10
		self.G = 6.67408e-8
		self.pi = np.pi

		# angular size
		self.deg = np.pi/180
		self.arcmin = self.deg/60
		self.arcsec = self.deg/3600

		# sterian
		self.sqdeg = (np.pi/180)**2
		self.sqamin = self.sqdeg/3600
		self.sqasec = self.sqamin/3600

		return
