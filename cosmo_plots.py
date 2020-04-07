from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess

main_dir = '/oasis/tscc/scratch/cchoban/FIRE_2_0_or_h553_criden1000_noaddm_sggs_dust/'
names = ['Species', 'Elemental', 'Species_no_cutoff']
snap_dirs = [main_dir + i + '/output/' for i in names] 
halo_dirs = [main_dir + i + '//AHF_data/halos/' for i in names] 
image_dir = './cosmo_images/'
sub_dir = 'snapshot_images/' # subdirectory since this will make a lot of images
halo_name = 'halo_0000000.dat'

cosmological = True
implementations = ['species','elemental','species']

# First create ouput directory if needed
try:
    # Create target Directory
    os.mkdir(image_dir)
    print "Directory " + image_dir +  " Created " 
except:
    print "Directory " + image_dir +  " already exists"


# First and last snapshot numbers
startnum = 64
endnum = 598

# Maximum radius used for getting data
r_max= 5 # kpc

for i,snap_dir in enumerate(snap_dirs):
	halo_dir = halo_dirs[i]
	name = names[i]
	print(name)

	# Load in halohistory data for main halo. All values should be in code units
	halo_data = Table.read(halo_dir + halo_name,format='ascii')

	for num in range(startnum,endnum+1):
		print(num)

		H = readsnap(snap_dir, num, 0, header_only=1, cosmological=cosmological)
		G = readsnap(snap_dir, num, 0, cosmological=cosmological)

		xpos =  halo_data['col7'][num-1]*H['time']/H['hubble']
		ypos =  halo_data['col8'][num-1]*H['time']/H['hubble']
		zpos =  halo_data['col9'][num-1]*H['time']/H['hubble']
		rvir = halo_data['col13'][num-1]*H['time']/H['hubble']
		center = np.array([xpos,ypos,zpos])

		DZ_vs_r([G], [H], [center], [rvir], bin_nums=50, time=True, foutname=image_dir+sub_dir+name+'_DZ_vs_r_%03d.png' % num)

		coords = G['p']
		# coordinates within a sphere of radius 5 kpc
		in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(r_max,2.)

		# Make phase plot
		phase_plot(G,H,time=True,mask=in_sphere,foutname=image_dir+sub_dir+name+"_phase_plot_%03d.png" % num)
		# Make D/Z vs density plot
		DZ_vs_dens([G],[H],time=True,mask_list=[in_sphere],foutname=image_dir+sub_dir+name+"_DZ_vs_dens_%03d.png" % num)
		# Make D/Z vs Z plot
		DZ_vs_Z([G],[H],time=True,mask_list=[in_sphere],Zmin=1E-4, Zmax=1e0,foutname=image_dir+sub_dir+name+"_DZ_vs_Z_%03d.png" % num)
		
	# Create movie of images
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+name+'_phase_plot_%03d.png '+name+'_phase_plot.mp4'],shell=True) 
	os.system('cp '+image_dir+sub_dir+name+'_phase_plot.mp4'+' '+image_dir)
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+name+'_DZ_vs_dens_%03d.png '+name+'_DZ_vs_dens.mp4'],shell=True) 
	os.system('cp '+image_dir+sub_dir+name+'_DZ_vs_dens.mp4'+' '+image_dir)
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+name+'_DZ_vs_r_%03d.png '+name+'_DZ_vs_r.mp4'],shell=True)
	os.system('cp '+image_dir+sub_dir+name+'_DZ_vs_r.mp4'+' '+image_dir) 
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+name+'_DZ_vs_Z_%03d.png '+name+'_DZ_vs_Z.mp4'],shell=True) 
	os.system('cp '+image_dir+sub_dir+name+'_DZ_vs_Z.mp4'+' '+image_dir) 