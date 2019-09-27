from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os

halo_dir = './halos/'
snap_dir = './output/'
image_dir = './images/'
halo_name = 'halo_0000000.dat'

# First create ouput directory if needed
try:
    # Create target Directory
    os.mkdir(image_dir)
    print "Directory " + image_dir +  " Created " 
except:
    print "Directory " + image_dir +  " already exists"

# Load in halohistory data for main halo. All values should be in code units
halo_data = Table.read(halo_dir + halo_name,format='ascii')

# First and last snapshot numbers
startnum = 589
endnum = 589
"""
for num in range(startnum,endnum+1):

	print num

	H = readsnap(snap_dir, num, 0, header_only=1)
	G = readsnap(snap_dir, num, 0)

	xpos =  halo_data['col7'][num]
	ypos =  halo_data['col8'][num]
	zpos =  halo_data['col9'][num]
	rvir = halo_data['col13'][num]
	center = np.array([xpos,ypos,zpos])

	coords = G['p']
	# coordinates within a sphere of radius Rvir
	in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(rvir,2.)

	# Make phase plot
	phase_plot(G,H,time=True,mask=in_sphere,foutname=image_dir+"phase_plot_%03d.png" % num)
	plt.close()
	DZ_vs_dens(G,H,time=True,mask=in_sphere,foutname=image_dir+"DZ_vs_dens_%03d.png" % num)
	plt.close()
"""

# Now preload the time evolution data
compile_dust_data(snap_dir, foutname='data.pickle', mask=True, overwrite=True, halo_dir=halo_dir+halo_name, Rvir_frac = 1., startnum=startnum, endnum=endnum, implementation='elemental')