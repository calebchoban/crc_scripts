import numpy as np
import os

# coordinate transform via rotation
def coordinates_rotate(x, y, z, theta, phi):
    
    if (theta==0.) and (phi==0.):
        # do nothing to save compute expense
        x_new, y_new, z_new = x, y, z
    else:
        x_new = x*np.cos(phi) + y*np.sin(phi)
        y_new = -x*np.cos(theta)*np.sin(phi) + y*np.cos(theta)*np.cos(phi) + z*np.sin(theta)
        z_new = x*np.sin(theta)*np.sin(phi) - y*np.sin(theta)*np.cos(phi) + z*np.cos(theta)
    
    return x_new, y_new, z_new


# find appropriate scale bar and label
def find_scale_bar(L):

    if (L>=10000):
        bar = 1000.; label = '1 Mpc'
    elif (L>=1000):
        bar = 100.; label = '100 kpc'
    elif (L>=500):
        bar = 50.; label = '50 kpc'
    elif (L>=200):
        bar = 20.; label = '20 kpc'
    elif (L>=100):
        bar = 10.; label = '10 kpc'
    elif (L>=50):
        bar = 5.; label = '5 kpc'
    elif (L>=20):
        bar = 2.; label = '2 kpc'
    elif (L>=10):
        bar = 1.; label = '1 kpc'
    elif (L>=5):
        bar = 0.5; label = '500 pc'
    elif (L>=2):
        bar = 0.2; label = '200 pc'
    elif (L>=1):
        bar = 0.1; label = '100 pc'
    elif (L>0.5):
        bar = 0.05; label = '50 pc'
    elif (L>0.2):
        bar = 0.02; label = '20 pc'
    elif (L>0.1):
        bar = 0.01; label = '10 pc'
    elif (L>0.05):
        bar = 0.005; label = '5 pc'
    elif (L>0.02):
        bar = 0.002; label = '2 pc'
    elif (L>0.01):
        bar = 0.001; label = '1 pc'
    else:
        bar = 0.0005; label = '0.5 pc'

    return bar, label


# this is my routine to make image
def make_projected_image(p, wt, h=None, method='simple', **kwargs):
    
    # explicit keywords:
    ## h    - smoothing length
    ## method - 'count', 'simple' or 'smooth'
    #
    # other available keywords:
    ## cen  - center of the image
    ## L    - side length along x and y direction
    ## Lz   - depth along z direction, for particle trimming
    ## Nx   - number of pixels along x and y direction, default 250
    ## vmin - minimum scaling of the colormap
    ## vmax - maximum scaling of the colormap
    ## theta, phi - viewing angle
    
    # import library, set up plot
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from matplotlib.colors import LogNorm
    
    rc('axes', linewidth=3, labelsize=22)
    rc('xtick.major', size=8, width=2, pad=6)
    rc('xtick.minor', size=4, width=1, pad=6)
    rc('ytick.major', size=8, width=2, pad=4)
    rc('ytick.minor', size=4, width=1, pad=4)
    rc('legend', fontsize=22, numpoints=1, scatterpoints=1,
       frameon=False, labelspacing=0.2, handleheight=0.3,
       handletextpad=0.2, borderaxespad=0.2, columnspacing=0.5)
    rc('xtick', labelsize=22)
    rc('ytick', labelsize=22)
    rc('font', size=22, family='serif')
       
    # calculate center if one not given
    if 'cen' in kwargs:
        cen = kwargs['cen']
        xc, yc, zc = cen[0], cen[1], cen[2]
    else:
        xc = np.median(p[:,0])
        yc = np.median(p[:,1])
        zc = np.median(p[:,2])

    theta = kwargs['theta'] if 'theta' in kwargs else 0.
    phi = kwargs['phi'] if 'phi' in kwargs else 0.
    x, y, z = coordinates_rotate(p[:,0]-xc, p[:,1]-yc, p[:,2]-zc, theta, phi)
    
    # calculate L if one not given
    if 'L' not in kwargs:
        Lx = np.max(np.abs(x))
        Ly = np.max(np.abs(y))
        Lz = np.max(np.abs(z))
        L = max(Lx, Ly, Lz)
    else:
        L = kwargs['L']
    Lz = kwargs['Lz'] if 'Lz' in kwargs else L
    
    # trim the particle, set the boundary for particle deposit
    ok = (x>-L) & (x<L) & (y>-L) & (y<L) & (z>-Lz) & (z<Lz)
    x, y, wt = x[ok], y[ok], wt[ok]
    if (method=='smooth') and (h is not None): h = h[ok]

    # set boundaries and grid size
    Nx = kwargs['Nx'] if 'Nx' in kwargs else 250
    dx = 2.0*np.float(L)/np.float(Nx)
    le, re, N = [-L,-L], [L,L], [Nx,Nx]

    # now, call the routine
    import deposit
    H = deposit.deposit(x, y, h=h, wt=wt, le=le, re=re, N=N, method=method)
    H /= (1.0e-4*dx*dx) # in Msun per pc^2
    vmax = kwargs['vmax'] if 'vmax' in kwargs else 1.1*np.max(H)
    vmin = kwargs['vmin'] if 'vmin' in kwargs else max(vmax/1.0e4,min(np.min(H[H>0])/1.1,vmax/100.))
                                       
    # set up the plot
    fig = plt.figure(0, (8.8,10))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor('k')
    ax = fig.add_axes([0,0.12,1,0.88])
    ax.patch.set_facecolor('k')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax_cbar = fig.add_axes([0.25,0.10,0.50,0.02])
    ax_cbar.tick_params(axis='x', colors='w', size=0)
                                       
    p = ax.imshow(H.transpose(), origin='lower', extent=(-L,L,-L,L), interpolation='bicubic',
                  norm=LogNorm(), cmap='gnuplot2', vmin=vmin, vmax=vmax)
    ax.set_xlim(-L,+L)
    ax.set_ylim(-L,+L)
    cbar = plt.colorbar(p, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(r"$\mathrm{M_{\odot}\,pc^{-2}}$", color='w')
                                       
    # add scale bar
    bar, label = find_scale_bar(L)
    ax.plot([-0.7*L-bar/2,-0.7*L+bar/2], [-0.87*L,-0.87*L], '-', c='w', lw=2)
    ax.annotate(label, (0.15,0.05), xycoords='axes fraction', color='w', ha='center', va='top')
                                       
    # add time label
    if 'time' in kwargs:
        ax.annotate(kwargs['time'], (0.97,0.97), xycoords='axes fraction', color='w', ha='right', va='top')
                                       
    plt.show()
    
    return H


# this calls PFH visualization routine
def image_maker(sp, ptype, **kwargs):
    
    # available keywords
    ## cen  - center of the image
    ## L    - side length along x and y direction
    ## Lz   - depth along z direction, for particle trimming
    ## idir - image of the directory
    ## fbase - file name base of the image
    #
    # many keywords from PFH image_maker routine, see details below
    ## theta, phi - viewing angle
    ## maxden, dynrange - color scaling of the image

    if (sp.k==-1): return -1 # no snapshot
    
    # derived keywords that will be sent to the routine
    center = kwargs['cen'] if 'cen' in kwargs else [0.,0.,0.]
    L = kwargs['L'] if 'L' in kwargs else 10.0
    Lz = kwargs['Lz'] if 'Lz' in kwargs else L
    xrange, yrange, zrange = [-L,L], [-L,L], [-Lz,Lz]
    show_gasstarxray = 'gas' if ptype==0 else 'star'
    idir = kwargs['idir'] if 'idir' in kwargs else sp.sdir+"/images"
    if (not os.path.exists(idir)): os.system("mkdir -p "+idir)
    fbase = kwargs['fbase'] if 'fbase' in kwargs else "snap%03d" %sp.snum
    filename = idir + "/" + fbase + "_" + show_gasstarxray
    cosmo = 1 if sp.cosmological==1 else 0
    
    # these keywords are taken from PFH image_maker routine
    # expand this is we use more features in the future
    theta = kwargs['theta'] if 'theta' in kwargs else 0.0
    phi = kwargs['phi'] if 'phi' in kwargs else 0.0
    dynrange = kwargs['dynrange'] if 'dynrange' in kwargs else 1.0e2
    maxden = kwargs['maxden'] if 'maxden' in kwargs else 0.0
    maxden_rescale = kwargs['maxden_rescale'] if 'maxden_rescale' in kwargs else 1.0
    do_with_colors = kwargs['do_with_colors'] if 'do_with_colors' in kwargs else 1
    threecolor = kwargs['threecolor'] if 'threecolor' in kwargs else 1
    nasa_colors = kwargs['nasa_colors'] if 'nasa_colors' in kwargs else 1
    sdss_colors = kwargs['sdss_colors'] if 'sdss_colors' in kwargs else 0
    project_to_camera = kwargs['project_to_camera'] if 'project_to_camera' in kwargs else 0
    include_lighting = kwargs['include_lighting'] if 'include_lighting' in kwargs else 0
    
    # construct the keywords that will be send to the routine
    keys = {'xrange':xrange, 'yrange':yrange, 'zrange':zrange,
            'center':center, 'cosmo':cosmo, 'theta':theta, 'phi':phi,
            'show_gasstarxray':show_gasstarxray,
            'set_filename':1, 'filename':filename,
            'snapdir_master':"", 'outdir_master':"",
            'dynrange':dynrange, 'maxden':maxden,
            'maxden_rescale':maxden_rescale,
            'do_with_colors':do_with_colors, 'threecolor':threecolor,
            'nasa_colors':nasa_colors, 'sdss_colors':sdss_colors,
            'project_to_camera':project_to_camera,
            'include_lighting':include_lighting}
    
    # call the routine
    from pfh.visualization.image_maker import image_maker as imaker
    image24, massmap = imaker(sp.sdir, sp.snum, **keys)
            
    return massmap


# make threeband image from massmap
def make_threeband_image(fbase, itype='star', L=10.0, tlabel='', maxden=0.01, dynrange=200, save=False):

    # import library, set up plot
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font', size=30, family='serif')

    image = return_threeband_image(fbase, itype=itype, maxden=maxden, dynrange=dynrange)
    
    # set up the plot
    fig = plt.figure(0, (8,8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    p = ax.imshow(image, origin='lower', extent=(-L,L,-L,L), interpolation='bicubic')
    ax.set_xlim(-L,+L)
    ax.set_ylim(-L,+L)
                  
    # add scale bar
    bar, label = find_scale_bar(L)
    ax.plot([-0.76*L-bar/2,-0.76*L+bar/2], [-0.81*L,-0.81*L], '-', c='w', lw=3)
    ax.annotate(label, (0.12,0.08), xycoords='axes fraction', color='w', ha='center', va='top')
                  
    # add time label
    ax.annotate(tlabel, (0.12,0.92), xycoords='axes fraction', color='w', ha='center', va='bottom')
    if (save): plt.savefig(fbase+'.jpg', format='jpg')
    plt.show()

    return


# return threeband image from massmap
def return_threeband_image(fbase, itype='star', maxden=0.01, dynrange=200):
    
    import h5py
    import pfh.visualization.make_threeband_image as mti
    
    # read massmap data
    f = h5py.File(fbase+'.hdf5', 'r')
    massmap = f["massmap"][...]
    r, g, b = massmap[:,:,0], massmap[:,:,1], massmap[:,:,2]
    f.close()
    
    image24, cmap = mti.make_threeband_image_process_bandmaps(r, g, b, maxden=maxden, dynrange=dynrange)
    if (itype=='gas'):
        image = mti.layer_band_images(image24, cmap)
    else:
        image = image24
    
    return image
