from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess

halo_dir = '/oasis/tscc/scratch/cchoban/FIRE_2_0_or_h553_criden1000_noaddm_sggs_dust/Elemental/AHF_data/halos/'
snap_dir = '/oasis/tscc/scratch/cchoban/FIRE_2_0_or_h553_criden1000_noaddm_sggs_dust/Elemental/output/'
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
startnum = 70
endnum = 598
"""


for num in range(startnum,endnum+1):

	print num

	H = readsnap(snap_dir, num, 0, header_only=1)
	G = readsnap(snap_dir, num, 0)

	xpos =  halo_data['col7'][num-1]
	ypos =  halo_data['col8'][num-1]
	zpos =  halo_data['col9'][num-1]
	rvir = halo_data['col13'][num-1]
	center = np.array([xpos,ypos,zpos])

	DZ_vs_r(G, H, center, rvir, bin_nums=50, time=True, foutname=image_dir+'DZ_vs_r%03d.png' % num)

	coords = G['p']
	# coordinates within a sphere of radius Rvir
	in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(rvir,2.)

	# Make phase plot
	phase_plot(G,H,time=True,mask=in_sphere,foutname=image_dir+"phase_plot_%03d.png" % num)
	plt.close()
	# Make D/Z vs density plot
	DZ_vs_dens(G,H,time=True,mask=in_sphere,foutname=image_dir+"DZ_vs_dens_%03d.png" % num)
	plt.close()
	"""
"""
# Create movie of images
subprocess.call(['./movie_maker.sh ' + image_dir + ' ' + str(startnum) + ' 25 phase_plot_%03d.png phase_plot.mp4'],shell=True) 
subprocess.call(['./movie_maker.sh ' + image_dir + ' ' + str(startnum) + ' 25 DZ_vs_dens_%03d.png DZ_vs_dens.mp4'],shell=True) 
"""
#subprocess.call(['./movie_maker.sh ' + image_dir + ' ' + str(startnum) + ' 25 DZ_vs_r_%03d.png DZ_vs_r.mp4'],shell=True) 


# Now preload the time evolution data
compile_dust_data(snap_dir, foutname='data.pickle', mask=True, overwrite=True, halo_dir=halo_dir+halo_name, Rvir_frac = 1., startnum=10, endnum=599, implementation='elemental')

#DZ_vs_time(finname='data.pickle', data_dir='data/')
