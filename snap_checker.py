import os
import subprocess
from gizmo_library.time_evolution import Dust_Evo
from gizmo import *
from dust_plots import *


dirc = ['/oasis/tscc/scratch/cchoban/test/']
snap_dirs = ['/oasis/tscc/scratch/cchoban/test/output/']


implementation = 'species'

cosmological = False
pb_fix=True
dust_depl=False

labels = ['Species Fiducial']

# List of snapshots to compare
snaps = [228]

# Maximum radius, disk, height, and disk orientation used for getting data
r_max = 40 # kpc
disk_height = 4 # kpc

for i, num in enumerate(snaps):
	print("Snapshot Number:",num)
	galaxies = []

	for j,snap_dir in enumerate(snap_dirs):
		print snap_dir
		galaxy = load_disk(snap_dir, num, cosmological=cosmological, id=-1, mode='AHF', hdir=None, periodic_bound_fix=pb_fix, rmax=r_max, height=disk_height)
		galaxies += [galaxy]
	

	print galaxy.sp.Flag_DustSpecies
	print galaxy.sp.Flag_DustMetals

	G = galaxy.loadpart(0)
	nH = G.rho*config.UnitDensity_in_cgs * ( 1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS


	print "Num Nans:",len(G.dz[:,0][np.isnan(G.dz[:,0])])
	print len(G.dz[np.isnan(G.dz)])
	print len(G.dzs[np.isnan(G.dzs)])
	print len(G.spec[np.isnan(G.spec)])
	print len(G.z[np.isnan(G.z)])
	nan_ind = np.argwhere(np.isnan(G.dz[:,0]))
	print "Nan nH:", nH[nan_ind]
	print "ID:", G.id[nan_ind]

	print G.spec[nan_ind]
	print G.dz[nan_ind]
	print G.dzs[nan_ind]


	print len(np.any(G.dz<0,axis=1))
	print G.dz[np.any(G.dz<0,axis=1)]
	print G.spec[np.any(G.dz<0,axis=1)]
	print G.dzs[np.any(G.dz<0,axis=1)]
	print "Negative Numbers G.dz:", len(G.dz[G.dz<0])
	print "Negative Numbers G.spec:", len(G.spec[G.spec<0])
	print "Negative Numbers G.dzs:", len(G.dzs[G.dzs<0])

	print "Max index"
	max_ind = np.nanargmax(G.dz[:,0])
	print np.nanmax(np.sum(G.spec, axis=1))
	print np.nanmax(G.dz[:,0])

	print G.spec[max_ind]
	print G.dz[max_ind]
	print G.dzs[max_ind]
	print G.spec[max_ind,-1]+G.spec[max_ind,3]
	print G.dz[max_ind,10]
	print "nH:",nH[max_ind]

	print G.dz[max_ind,0]/G.z[max_ind,0]
	print G.dz[max_ind]/G.z[max_ind,:11]

	print "Max Sil"
	max_sil = np.argmax(G.spec[:,0])
	print "nH:",nH[max_sil]
	print np.sum(G.spec[max_sil])
	print G.spec[max_sil]
	print G.dz[max_sil]
	print G.dz[max_sil,4]+G.dz[max_sil,6]+G.dz[max_sil,7] - (G.spec[max_sil,2]*28.1/(28.1+12))-G.spec[max_sil,4]
	print G.spec[max_sil,-1]+G.spec[max_sil,3]
	print G.dz[max_sil,0]/G.z[max_sil,0]
	print G.dz[max_sil]/G.z[max_sil,:11]



	print "Top DZ check"
	top_idxs = np.argsort(G.dz[:,0]/G.z[:,0])[-10:]
	#top_idxs = np.argsort(nH)[-10:]
	for k, idx in enumerate(top_idxs):
		print k
		print '\tTotal D/Z:', G.dz[idx,0]/G.z[idx,0]
		nH = G.rho*config.UnitDensity_in_cgs * ( 1. - (G.z[:,0]+G.z[:,1])) / config.H_MASS
		print "\tnH:",nH[idx], "T:", G.T[idx]
		print '\tDZ:', G.dz[idx]
		print '\tZ:', G.z[idx,:11]
		print '\tD/Z:', G.dz[idx]/G.z[idx,:11]
		print '\tTotal DZ check:', np.sum(G.dz[idx,1:]), G.dz[idx,0], np.sum(G.spec[idx])
		print '\tSpec:', G.spec[idx]
		print '\tMol.:', G.dust_mol[idx]


	print "Max O"
	i_max = np.argmax(G.spec[:,4])
	print "nH:",nH[i_max], "T:", G.T[i_max]
	print G.dz[i_max]/G.z[i_max,:11]
	print G.spec[i_max,4], G.z[i_max,4]-(G.spec[i_max,2]*28.1/(28.1+12))
	print np.sum(G.spec*G.m[:,np.newaxis]*1E10,axis=0)
	print np.sum(G.dz[:,0])/np.sum(G.z[:,0])
	print np.sum(G.spec)/np.sum(G.z[:,0])
	print np.sum(G.dz[:,0,np.newaxis]*G.dzs*G.m[:,np.newaxis]*1E10,axis=0)
	print G.dust_mol[i_max]

	"""
	if mol_param=='fH2':
		dmol = G.dust_mol[:,0]
	elif mol_param=='fMC':
		dmol = G.dust_mol[:,1]
	elif mol_param=='CinCO':
		dmol = G.dust_mol[:,2]/G.z[:,2]
	"""
	dmol_vs_params('fH2', ['nH', 'T'], galaxies, bin_nums=50, time=None, labels=None, foutname='snap_'+str(num)+'_fH2_vs_nH_T.png', std_bars=True, style='color')
	dmol_vs_params('fMC', ['nH', 'T'], galaxies, bin_nums=50, time=None, labels=None, foutname='snap_'+str(num)+'_fMC_vs_nH_T.png', std_bars=True, style='color')
	dmol_vs_params('CinCO', ['nH', 'T'], galaxies, bin_nums=50, time=None, labels=None, foutname='snap_'+str(num)+'_CinCO_vs_nH_T.png', std_bars=True, style='color')

	DZ_vs_params(['nH'], galaxies, bin_nums=40, time=None, labels=labels, foutname='snap_'+str(num)+'_DZ_vs_nH.png', std_bars=True, style='color', include_obs=True)

	elems = ['Mg','Si','Fe','O','C']
	elem_depletion_vs_param(elems, 'nH', galaxies, bin_nums=50, time=None, labels=labels, foutname='snap_'+str(num)+'_obs_elemental_dep_vs_dens.png', \
						std_bars=True, style='color', include_obs=True)
