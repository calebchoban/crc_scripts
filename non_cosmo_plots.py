from readsnap import readsnap
from dust_plots import *
from astropy.table import Table
import os
import subprocess



main_dir = '/oasis/tscc/scratch/cchoban/non_cosmological_runs/Species/'
names = ['fiducial_model','elem_creation_eff','enhanced_acc','extra_O']
snap_dirs = [main_dir + i + '/output/' for i in names] 
labels = ['Fiducial','Spec. Creation Eff.','Enhanced Dest.','Enhanced O']
image_dir = './non_cosmo_species_images/'
sub_dir = 'snapshot_images/' # subdirectory since this will make a lot of images

implementation = 'species'

cosmological = False

# First create ouput directory if needed
try:
    # Create target Directory
    os.mkdir(image_dir)
    print "Directory " + image_dir +  " Created " 
except:
    print "Directory " + image_dir +  " already exists"
try:
    # Create target Directory
    os.mkdir(image_dir + sub_dir)
    print "Directory " + image_dir + sub_dir + " Created " 
except:
    print "Directory " + image_dir + sub_dir + " already exists"


# First and last snapshot numbers
startnum = 0
endnum = 400

# Maximum radius used for getting data
r_max_phys = 20 # kpc

for i,snap_dir in enumerate(snap_dirs):
	name = names[i]
	print(name)

	for num in range(startnum,endnum+1):
		print(num)

		H = readsnap(snap_dir, num, 0, header_only=1, cosmological=cosmological)
		G = readsnap(snap_dir, num, 0, cosmological=cosmological)

		coords = G['p']
		# Recenter coords at center of periodic box
		boxsize = H['boxsize']
		mask1 = coords > boxsize/2; mask2 = coords <= boxsize/2
		coords[mask1] -= boxsize/2; coords[mask2] += boxsize/2; 
		center = np.average(coords, weights = G['m'], axis = 0)
		DZ_vs_r([G], [H], [center], [r_max_phys], bin_nums=50, time=True, foutname=image_dir+sub_dir+implementation+'_'+name+'_DZ_vs_r_%03d.png' % num,cosmological=cosmological)

		# coordinates within a sphere of radius r_max_phys
		in_sphere = np.power(coords[:,0] - center[0],2.) + np.power(coords[:,1] - center[1],2.) + np.power(coords[:,2] - center[2],2.) <= np.power(r_max_phys,2.)

		# Make phase plot
		phase_plot(G,H,time=True,mask=in_sphere,foutname=image_dir+sub_dir+implementation+'_'+name+"_phase_plot_%03d.png" % num,cosmological=cosmological)
		# Make D/Z vs density plot
		DZ_vs_dens([G],[H],time=True,mask_list=[in_sphere],foutname=image_dir+sub_dir+implementation+'_'+name+"_DZ_vs_dens_%03d.png" % num,cosmological=cosmological)
		# Make D/Z vs Z plot
		DZ_vs_Z([G],[H],time=True,mask_list=[in_sphere],Zmin=1E0, Zmax=1e1,foutname=image_dir+sub_dir+implementation+'_'+name+"_DZ_vs_Z_%03d.png" % num,cosmological=cosmological)
		
	# Create movie of images
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+implementation+'_'+name+'_phase_plot_%03d.png '+implementation+'_'+name+'_phase_plot.mp4'],shell=True) 
	os.system('cp '+image_dir+sub_dir+implementation+'_'+name+'_phase_plot.mp4'+' '+image_dir)
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+implementation+'_'+name+'_DZ_vs_dens_%03d.png '+implementation+'_'+name+'_DZ_vs_dens.mp4'],shell=True) 
	os.system('cp '+image_dir+sub_dir+implementation+'_'+name+'_DZ_vs_dens.mp4'+' '+image_dir)
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+implementation+'_'+name+'_DZ_vs_r_%03d.png '+implementation+'_'+name+'_DZ_vs_r.mp4'],shell=True)
	os.system('cp '+image_dir+sub_dir+implementation+'_'+name+'_DZ_vs_r.mp4'+' '+image_dir) 
	subprocess.call(['./movie_maker.sh ' + image_dir + sub_dir + ' ' + str(startnum) + ' 25 '+implementation+'_'+name+'_DZ_vs_Z_%03d.png '+implementation+'_'+name+'_DZ_vs_Z.mp4'],shell=True) 
	os.system('cp '+image_dir+sub_dir+implementation+'_'+name+'_DZ_vs_Z.mp4'+' '+image_dir) 