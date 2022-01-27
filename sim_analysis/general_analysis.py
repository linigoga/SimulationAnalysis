import numpy as np
from openpmd_viewer import ParticleTracker
from openpmd_viewer.addons import LpaDiagnostics,pic
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import e,m_e,c
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,TransformedBbox, BboxPatch, BboxConnector 
import matplotlib.colors as colors
from scipy.interpolate import splev,splrep,CubicSpline

def smooth_trajectories(cache,ts):
    '''
    Function which yields back the smoothened trajectories using splines
    '''
    
    time = ts.t
    ct = c*time
    smooth_cache = {}
    
    keys = cache.keys()
    for key in tqdm(keys):
        _,x,y,z,px,py,pz,w = cache[key]
        
        X = 1e6*np.nan_to_num(x)
        CT = 1e6*ct[:-1][X != 0]
        Y = 1e6*np.array(y)[X != 0];Z = 1e6*np.array(z)[X != 0];W = np.array(w)[X != 0]
        Px = np.array(px)[X != 0]; Py = np.array(py)[X != 0];Pz = np.array(pz)[X != 0]
        X = X[X != 0]
        
        
        ti = np.linspace(CT[0],CT[-1],1000)

        tck = splrep(CT,Z,s=0)
        z_ = splev(ti,tck,der=0)

        tck = splrep(CT,X,s=0)#for x
        x_ = splev(ti,tck,der=0)

        tck = splrep(CT,Y,s=0)#for y
        y_ = splev(ti,tck,der=0)
        
        tck = splrep(CT,Px,s=0)#for px
        px_ = splev(ti,tck,der=0)
        
        tck = splrep(CT,Py,s=0)#for py
        py_ = splev(ti,tck,der=0)
        
        tck = splrep(CT,Pz,s=0)#for pz
        pz_ = splev(ti,tck,der=0)
        
        tck = splrep(CT,W,s=0)#for pz
        w_ = splev(ti,tck,der=0)
        
        
        smooth_cache[key] = [x_,y_,z_,px_,py_,pz_,w_]
        
    return smooth_cache




def check_cross(cache,method = 1):
    '''
    A basic function which is able to check if a particle crosses
    the back of the bubble. In order to feed it information,
    an array with four to 8 elements depending if the method is 3.
    
    returns True of False
    '''
    
    if method == 1:
        A1,A2,B1,B2 = cache
        a1,a2 = np.sign([A1,A2])
        b1,b2 = np.sign([B1,B2])
        
        if a1 != a2 and b1 != b2:
            #Method one compares position
            cross = True
        else:
            cross = False

    elif method == 2:
        #Method two compares momentum
        A1,A2,B1,B2 = cache
        a1,a2 = np.sign([A1,A2])
        b1,b2 = np.sign([B1,B2])
        
        if a1 == a2 and b1 == b2:
            cross = True
        else:
            cross = False

    elif method == 3:
        #Method three compares position and momentum
        try:
            A1,A2,B1,B2,C1,C2,D1,D2 = cache
        except ValueError:
            print(f'cache has not enough values to unpack, expected 8, got {np.shape(cache)[0]}')
            raise

        a1,a2 = np.sign([A1,A2])
        b1,b2 = np.sign([B1,B2])
        c1,c2 = np.sign([C1,C2])
        d1,d2 = np.sign([D1,D2])

        
        if a1 != a2 and b1 != b2 and c1 == c2 and d1 == d2:
            cross = True

        else:
            cross = False
    else:
        print('there are 3 methods, choose from 1 to 3 to perform the check.')
        return
    
    return cross

def get_crossingPoint(x,y,ts):
    '''
    Gets where particles are crossing by taking the trajectories and taking the zeros. Returns the index at where the points have crossed the origin 
    
    
    Input
    -----------------------------------
    x: Array containing the x positions along the simulation
    y: Array containing the y positions along the simulation
    ts: Object corresponding the the openPMD package
    
    Returns 
    -----------------------------------
    
    idx: int, index corresponding to where they particle has crossed the x axis
    idy: int, index corresponding to where they particle has crossed the y axis
    
    '''
    time = ts.t
    ct = c*time
    
    
    
    X = 1e6*np.nan_to_num(x)
    Y = 1e6*np.array(y)[X != 0]
    CT = 1e6*ct[:-1][X != 0]
    X = X[X != 0]

    cross = (np.sign(X[0]) != np.sign(X[-1])) & (np.sign(Y[0]) != np.sign(Y[-1]))
    
    if cross:
        ti = np.linspace(CT[0],CT[-1],len(X))
        indices = np.indices(np.shape(ti)).ravel()

        tck = splrep(CT,X,s=0)
        x2_root_ind = [indices[np.argmin(np.abs(ti - roots))] for roots in sproot(tck)]
        x2 = splev(ti,tck,der=0)
        
        
        #print(1e6*np.nan_to_num(x),x2[x2_root_ind[-1]])
        idx = np.argmin(np.abs(1e6*np.nan_to_num(x) - x2[x2_root_ind]))

        tck = splrep(CT,Y,s=0)
        y2_root_ind = [indices[np.argmin(np.abs(ti - roots))] for roots in sproot(tck)]
        y2 = splev(ti,tck,der=0)
        idy = np.argmin(np.abs(1e6*np.nan_to_num(y) - y2[y2_root_ind]))
    
    
        return idx,idy
    
    return np.nan,np.nan
    
def consistency_cross_check(x,y,z,px,py,pz,w,method = 1):
    
    
    
    cache = {}
    
    for i,(X,Y,Z,Px,Py,Pz,W) in enumerate(zip(x,y,z,px,py,pz,w)):
        ind_x = np.indices(np.shape(X)).ravel()
    
        XX = np.nan_to_num(X)
        indx = ind_x[XX != 0]

        for j in range(len(indx) - 1):
            diffx = indx[j+1] - indx[j]
            
            if diffx != 1:
                print(f'{i} and {j} are fucked')
        A1,A2 = Y[indx[0]],X[indx[-1]]
        B1,B2 = Y[indx[0]],Y[indx[-1]]
        C1,C2 = Px[indx[0]],Px[indx[-1]]
        D1,D2 = Py[indx[0]],Py[indx[-1]]
        if method == 1:
            arr = [A1,A2,B1,B2]
        elif method == 2:
            arr = [C1,C2,D1,D2]
        elif method == 3:
            arr = [A1,A2,B1,B2,C1,C2,D1,D2]
        else:
            print('invalid value, select method from 1 to 3')
            return
        cross = check_cross(arr,method)
        cache[f'{i}'] = [cross,X,Y,Z,Px,Py,Pz,W]
    
    return cache


def localizeLongestSelection(dictionary,keys):
    longest = 0 
    longest_key = 0
    iterations = 0
    for key in keys:
        cross,x,y,z,px,py,pz,w = dictionary[key]
        X = np.nan_to_num(x)
        idx = np.indices(np.shape(X)).ravel()
        idx = idx[X != 0]
        X = X[X != 0]
        length = len(X)



        if length > longest:
            longest = length
            longest_key = key
            iterations = idx
    
    
    return longest,longest_key,iterations

def scatter_formater(x):
    x = np.nan_to_num(x)
    x = x[x != 0]
    return x


def load_tracers(iteration,select_trace,PATH):
    """
    MEMORY LEAK, A more convenient way to use this is by taking the transpose of the obtained values
    """
    ts = OpenPMDTimeSeries(PATH)
    pt_tracers = ParticleTracker(ts,iteration=iteration,select=select_trace,
                     species='tracers',
                     preserve_particle_index=True)
    x_tracers,y_tracers,z_tracers,px_tracers,py_tracers,pz_tracers,w_tracers = ts.iterate(ts.get_particle,
                                                                            ['x','y','z','ux','uy','uz','w'],
                                                                            select = pt_tracers,
                                                                            species='tracers'
                                                                              )
    shape = x_tracers[1].shape

    xt_t = []
    yt_t = []
    zt_t = []
    px_t = []
    py_t = []
    pz_t = []
    wt_t = []



    for i in range(shape[0]):
        values_x = []
        values_y = []
        values_z = []
        values_px = []
        values_py = []
        values_pz = []
        values_w = []
        #for x,y,z,px,py in zip(x_tracers[b:a],y_tracers[b:a],z_tracers[b:a],px_tracers[b:a],py_tracers[b:a]):
        for x,y,z,px,py,pz,w in zip(x_tracers,y_tracers,z_tracers,px_tracers,py_tracers,pz_tracers,w_tracers):
            if len(x) != 0:
                values_x.append(x[i])
                values_y.append(y[i])
                values_z.append(z[i])
                values_px.append(px[i])
                values_py.append(py[i])
                values_pz.append(pz[i])
                values_w.append(w[i])

        xt_t.append(values_x)
        yt_t.append(values_y)
        zt_t.append(values_z)
        px_t.append(values_px)
        py_t.append(values_py)
        pz_t.append(values_pz)
        wt_t.append(values_w)
    
    return xt_t,yt_t,zt_t,px_t,py_t,pz_t,wt_t


def load_tracers2(iteration,select_trace,PATH):
    """
    Tentative Replacement to load_tracers()
    """
    ts = OpenPMDTimeSeries(PATH)
    pt_tracers = ParticleTracker(ts,iteration=iteration,select=select_trace,
                     species='tracers',
                     preserve_particle_index=True)
    x_tracers,y_tracers,z_tracers,px_tracers,py_tracers,pz_tracers,w_tracers = ts.iterate(ts.get_particle,
                                                                            ['x','y','z','ux','uy','uz','w'],
                                                                            select = pt_tracers,
                                                                            species='tracers'
                                                                              )
    xt_t = np.transpose(x_tracers[1:])
    yt_t = np.transpose(y_tracers[1:])
    zt_t = np.transpose(z_tracers[1:])
    px_t = np.transpose(px_tracers[1:])
    py_t = np.transpose(py_tracers[1:])
    pz_t = np.transpose(pz_tracers[1:])
    wt_t = np.transpose(w_tracers[1:])
    
    return xt_t,yt_t,zt_t,px_t,py_t,pz_t,wt_t

def calc_signal_fft(x,y,bins = 250):
    bins = bins
    
    a = np.linspace(-np.pi,np.pi,bins)
    angles = np.arctan2(y,x)
    
    az_dist = np.zeros_like(a)
    index_angles = np.digitize(angles,a[:-1])
    
    for idx in index_angles:
        az_dist[idx] += 1
        
    da = a[1]-a[0]
    freq = 2*np.pi*np.fft.fftfreq(len(az_dist), da)
    fmax = int(len(freq)/2)
    freq = freq[1:fmax]
    fft = np.abs(np.fft.fft(az_dist))[1:fmax]
    
    
    return a,az_dist,freq,fft



def check_particleCrossPoint_2(x,y,z,px,py,pz):
    '''
    Checks where the particles may have crossed by iteratively checking if both
    x and y have changed signs. 
    ______________________________________________
    Input: 
        3 arrays of equal length containing trajectories in x, y and z.
    
    _______________________________________________
    Output:
        3 arrays with two members pertaining to the before and after spot of crossing the center
        as well as a variable assigned as True/False yielding if a particle crossed.
    
    '''
    
    X,Y,Z = 1e6*np.array(x),1e6*np.array(y),1e6*np.array(z)
    XX = np.nan_to_num(X)
    ind_x = np.indices(np.shape(XX)).ravel()
    ind_x = ind_x[XX != 0]
    prev = ind_x[0]
    cross = False
    for i in range(len(ind_x) - 1):
        
        new = prev+1
        diff_x = X[prev]/X[new]
        diff_y = Y[prev]/Y[new]
        
        if diff_x < 0 and diff_y < 0:
            #print(diff_x,diff_y,X[prev],X[new])
            #print(f'crossing at x:{X[prev],X[new]},y:{Y[prev],Y[new]},z:{Z[prev]}')
            cross = True
            return X[prev:new+1],Y[prev:new+1],Z[prev:new+1],px[prev:new+1],py[prev:new+1],pz[prev:new+1],cross
            break
        prev += 1
    if not cross:
        return [[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],cross]
    
    

    
def plotTrajectories(dict_,rand_cross,location,iterations,PATH):
    '''
    Routine to plot the trajectories of the particles with dynamic colouring,
    scatterplots with colourmaps denoting the momenta of the particles in the 
    radial and longitudinal momenta. One can plot the histograms demonstrating
    the momentum distributions in the radial and longitudinal momenta as well
    as the zoomed in version of the transverse plane and the longitudinal plane
    with the number density superimposed with the longitudinal trajectories. 
    ___________________________________________________________________________
    Inputs:
    
        dict_: 
            Dict type. Dictionary containing the particle information
        rand_cross: 
            Array type. Array containing the randomly selected keys to be plotted
        location:
            String type. Direction where the images will be saved.
        iterations:
            Array type. Iterations where the particles are saved.
        PATH:
            String type. Location of the  hdf5 files.
    
    '''
    ts = OpenPMDTimeSeries(PATH)
    
    
    plot_hist = input('plot histograms or zoom in a lower row, or plot all?')
    plot_ = plot_hist[:4].lower()
    if plot_ == 'hist':
        print('plotting hist')
    elif plot_  == 'zoom':
        print('plotting zoom')
    elif plot_  == 'all':
        print('plotting all')

    

    init_iter = iterations[0]
    ne,info = ts.get_field('rho',iteration=ts.iterations[init_iter])
    ne /= -e
    
    for ix in tqdm(iterations):
        #prepare the lists so we can plot the scatterplot
        XX,YY,ZZ,E =[],[],[],[]
        Px,Py,Pz = [],[],[]
        weights = []
        



        #Check what to plot
        if plot_ == 'hist':
            fig = plt.figure(figsize=(12,6))
            location = location
            try:
                os.makedirs(location)
            except OSError:
                if not os.path.join(location):
                    raise


            ax1 = fig.add_subplot(221);ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223);ax4 = fig.add_subplot(224)

        elif plot_ == 'zoom':
            
            location = location
            try:
                os.makedirs(location)
            except OSError:
                if not os.path.join(location):
                    raise

            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(221);ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223);ax4 = fig.add_subplot(224)

        elif plot_ == 'all':
            
            location = location
            try:
                os.makedirs(location)
            except OSError:
                if not os.path.join(location):
                    raise
            
            fig = plt.figure(figsize=(12,10))
            ax1 = fig.add_subplot(321);ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323);ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325);ax6 = fig.add_subplot(326)
            
            
            

        else:
            location = location
            try:
                os.makedirs(location)
            except OSError:
                if not os.path.join(location):
                    raise
            ax1 = fig.add_subplot(121);ax2 = fig.add_subplot(122)


        for key in rand_cross:
            cross,x,y,z,px,py,pz,w = dict_[key]
            X,Y,Z,crossed = check_particleCrossPoint(x,y,z)
            xx,yy = 1e6*np.array(x)[init_iter:ix+1],1e6*np.array(y)[init_iter:ix+1]
            zz = 1e6*np.array(z)[init_iter:ix+1]
            #print(x)
            # setting the arrays for the scatterplot
            XX.append(x[ix]);YY.append(y[ix]);ZZ.append(z[ix])
            Px.append(px[ix]);Py.append(py[ix]),Pz.append(pz[ix])
            weights.append(w[ix])

            if X[0] in xx and Y[0] in yy:
                color = sns.color_palette('Set2')[0]
            else:
                color = sns.color_palette('Set2')[1]
            ax1.plot(yy,xx,c=color,alpha = 0.5,linewidth=0.5)
            ax2.plot(zz,xx,c=color,alpha = 0.5,linewidth=0.5)


            if plot_ == 'zoom' or plot_ == 'all':
                
                ne,info = ts.get_field('rho',iteration=ts.iterations[ix])
                ne /= -e
                
                dr,dz = info.dr,info.dz
                R,Z_ = 1e6*info.r,1e6*info.z
                
                
                ax3.plot(yy,xx,c=color,alpha = 0.5)
                ax3.set_xlim(-0.3,0.3)
                ax3.set_ylim(-0.3,0.3)
                ax3.set_xticks(R,minor=True)
                ax3.set_yticks(R,minor=True)
                ax3.grid(which='minor',alpha = 1)

                
                im = ax4.imshow(ne,extent = 1e6*info.imshow_extent,vmax = 0.5e26,aspect='auto')
                cbar_ne = fig.colorbar(im,make_axes_locatable(ax4).append_axes('right',size='3%',pad=0.0))
                cbar_ne.set_label('$n_e, m^{-3}$')
                
                ax4.plot(zz,xx,c=color,alpha = 0.5,linewidth = 0.5)





        #scatterplot stuff
        tst = np.nan_to_num(XX)
        W = np.nan_to_num(weights)[tst != 0]
        XX_ = scatter_formater(XX)
        YY_ = scatter_formater(YY)
        ZZ_ = scatter_formater(ZZ)
        Px_ = scatter_formater(Px)
        Py_ = scatter_formater(Py)
        Pz_ = scatter_formater(Pz)
        Pr = (Px_*XX_ + Py_*YY_)/np.sqrt(XX_**2 + YY_**2)
        E_arr = np.sqrt(1 + Pr**2)


        #scatterplot 1
        color_var = Pr#E_arr
        alpha_var = W
        cmap = plt.cm.plasma
        min_alpha = 0.1
        alpha = (1-min_alpha)*alpha_var/alpha_var.max() + min_alpha
        colour = cmap(colors.Normalize(color_var.min(), color_var.max())(color_var) )
        colour[..., -1] = alpha

        cm2 = plt.cm.ScalarMappable(cmap='plasma')
        cm2.set_array(Pr) #cm2.set_array(E_arr)
        #cm2.set_clim(-2,2)

        ax1.scatter(1e6*np.array(YY_),1e6*np.array(XX_),c=colour,edgecolors='none', s=20,vmin=-2.0,vmax=2.0)
        cbar1 = fig.colorbar(cm2,ax=ax1,pad=0.0)
        cbar1.set_label('$p_r, m_e c$')

        #cbar1.ax.set_xlim(-2,2)

        #scatterplot 2

        color_var2 = Pz_
        alpha_var2 = alpha_var
        min_alpha = 0.1
        alpha2 = (1-min_alpha)*alpha_var2/alpha_var2.max() + min_alpha
        colour2 = cmap( colors.Normalize(color_var2.min(), color_var2.max())(color_var2) )
        colour2[..., -1] = alpha2
        cm3 = plt.cm.ScalarMappable(cmap='plasma')
        cm3.set_array(Pz_) #cm2.set_array(E_arr)
        #cm3.set_clim(-1.0,4.0)

        ax2.scatter(1e6*np.array(ZZ_),1e6*np.array(XX_),c=colour,edgecolors='none', s=20,vmin=-1.0,vmax=4.0)
        cbar2 = fig.colorbar(cm3,ax=ax2,pad=0.0)
        cbar2.set_label('$p_z, m_e c$')

        if plot_ == 'hist':
            #plot the histograms with the radial and longitudinal planes
            ax3.hist(Pr,bins=20)
            ax3.set_xlabel('$p_r, m_e c$')
            ax3.set_xlim(-2,2)


            ax4.hist(Pz_,bins = 20)
            ax4.set_xlabel('$p_z, m_e c$')
            ax4.set_xlim(-1,4)

        elif plot_ == 'zoom' or plot_ == 'all':
            
            # Plot the scattered points coloured with the radial and longitudinal momenta in the long plane and trans plane

            ax3.scatter(1e6*np.array(YY_),1e6*np.array(XX_),c=colour,edgecolors='none', s=20,vmin=-2.0,vmax=2.0)
            cbar3 = fig.colorbar(cm2,ax=ax3,pad=0.0)
            cbar3.set_label('$p_r, m_e c$')
            ax3.set_xlabel('$y, \mu m$')
            ax3.set_ylabel('$x, \mu m$')
            ax3.set_xlim(-0.3,0.3)
            ax3.set_ylim(-0.3,0.3)

            ax4.scatter(1e6*np.array(ZZ_),1e6*np.array(XX_),c=colour,edgecolors='none', s=10,vmin=-1.0,vmax=4.0)
            cbar4 =  fig.colorbar(cm2,ax = ax4, pad=0.1)#make_axes_locatable(ax4).append_axes('right',size='3%',pad=0.5))
            cbar4.set_label('$p_z, m_ec$')
            ax4.set_xlabel('$z, \mu m$')
            ax4.set_ylabel('$x, \mu m$')
            ax4.set_xlim(1e6*info.imshow_extent[0],1e6*info.imshow_extent[1])
            ax4.set_ylim(-10,10)

        elif plot_ == 'all':
            ax5.hist(Pr,bins=20)
            ax5.set_xlabel('$p_r, m_e c$')
            ax5.set_xlim(-2,2)


            ax6.hist(Pz_,bins = 20)
            ax6.set_xlabel('$p_z, m_e c$')
            ax6.set_xlim(-1,4)
            
            
            
        ax1.set_xlabel('$y, \mu m$')
        ax1.set_ylabel('$x, \mu m$')
        ax1.grid()
        ax1.set_xlim(-6,6)
        ax1.set_ylim(-6,6)

        ax2.set_xlabel('$z, \mu m$')
        ax2.set_ylabel('$x, \mu m$')
        ax2.grid()
        ax2.set_xlim(1e6*zpos[0]-10,1e6*zpos[0]+20)
        ax2.set_ylim(-6,6)



        plt.suptitle(f'{ts.iterations[ix]}',y=1.0)
        fig.tight_layout()

        fname = os.path.join(location,f'iter_{ix}.png')
        plt.savefig(fname,dpi = 200)

        plt.clf()
        plt.close('all')

    return

def field_transverse_plane(PATH,iteration,field = 'rho',coord = 'x',z_slice= 600e-6):
    
    T = LpaDiagnostics(PATH, check_all_files=False)
        
    if field == 'rho':
        field0R, info = T.get_field(field,iteration=iteration, m=0)
        field1R, _ = T.get_field(field,iteration=iteration, m=1, theta=0)
        field1I, _ = T.get_field(field,iteration=iteration, m=1, theta=np.pi/2)
        
        z_idx = np.abs(info.z - z_slice).argmin()
    
        field0R = -field0R[len(field0R)//2:,z_idx]/e
        field1R = -field1R[len(field1R)//2:,z_idx]/e
        field1I = -field1I[len(field1I)//2:,z_idx]/e
    else:
        field0R, info = T.get_field(field,coord = coord ,iteration=iteration, m=0)
        field1R, _ = T.get_field(field,coord = coord, iteration=iteration, m=1, theta=0)
        field1I, _ = T.get_field(field,coord = coord, iteration=iteration, m=1, theta=np.pi/2)
        
        z_idx = np.abs(info.z - z_slice).argmin()
        
        field0R = field0R[len(field0R)//2:,z_idx]
        field1R = field1R[len(field1R)//2:,z_idx]
        field1I = field1I[len(field1I)//2:,z_idx]
    
    r = info.r[len(info.r)//2:]
    #z_idx = np.abs(info.z - z_slice).argmin()
    
    field0R_spline = CubicSpline(r, field0R, extrapolate=False)
    field1R_spline = CubicSpline(r, field1R, extrapolate=False)
    field1I_spline = CubicSpline(r, field1I, extrapolate=False)
    # x,y limit to look at
    limit = r.max() #/np.sqrt(2)
    grid = np.linspace(start=-limit, stop=limit, num=2000)
    X, Y = np.meshgrid(grid, grid)
    R = np.sqrt(X**2 + Y**2)
    t = np.arctan2(Y,X)
    Field = field0R_spline(R) + field1R_spline(R)*np.cos(t) + field1I_spline(R)*np.sin(t)
    
    
    
    return Field,R,t,info

def select_particles(cache,r = 10,gamm=[1,10]):
    discrim_keys = []
    
    keys = cache.keys()
    
    for key in keys:
        _,x,y,z,px,py,pz,w = cache[key]
        #check if at the particles at the end are futher away from r threshold
        X = 1e6*np.nan_to_num(x)
        
        Y = 1e6*np.array(y)[X != 0];Z = 1e6*np.array(z)[X != 0]
        Px = np.array(px)[X != 0];Py =  np.array(py)[X != 0 ]; Pz = np.array(pz)[X != 0]
        X = X[X != 0]
        
        
        
        R = np.sqrt(X**2 + Y**2)
        gamma = np.sqrt(1 + Px**2 + Py**2 + Pz**2) 
        
        
        
        if R[-1] >= r and gamma[-1] > gamm[0] and gamma[-1] < gamm[-1]:
            discrim_keys.append(key)
        
        
    return discrim_keys

def get_info(iteration,x_lims,y_lims,z_lims):
    
    x,y,z,px,py,pz,w = ts.get_particle(['x','y','z','ux','uy','uz','w'],
                                       species='electrons',
                                       iteration=iteration,
                                       select ={'x':x_lims,
                                               'y':y_lims,
                                               'z':z_lims
                                              }
                                      )
    cache_parts = np.array([x,y,z,px,py,pz,w])
    
    ne,info = ts.get_field('rho',iteration=iteration)
    ne /= -e
    #E Fields
    Ex,_ = ts.get_field('E',coord='x',iteration=iteration)
    Ey,_ = ts.get_field('E',coord='y',iteration=iteration)
    Ez,_ = ts.get_field('E',coord='z',iteration=iteration)
    #B Fields
    Bx,_ = ts.get_field('B',coord='x',iteration=iteration)
    By,_ = ts.get_field('B',coord='y',iteration=iteration)
    Bz,_ = ts.get_field('B',coord='z',iteration=iteration)
    
    cache_fields = {}
    cache_fields['info'] = info
    cache_fields['ne'] = ne
    cache_fields['Ex'] = Ex; cache_fields['Ey'] = Ey; cache_fields['Ez'] = Ez
    cache_fields['Bx'] = Bx; cache_fields['By'] = By; cache_fields['Bz'] = Bz
    
    return cache_parts, cache_fields




def get_weightedColour(arr,weight):
    color_var = arr
    alpha_var = weight
    cmap = plt.cm.plasma
    min_alpha = 0.1
    alpha = (1-min_alpha)*alpha_var/alpha_var.max() + min_alpha
    if alpha.max() > 1:
        alpha[alpha > 1] = 1
    colour = cmap( colors.Normalize(color_var.min(), color_var.max())(color_var) )
    colour[..., -1] = alpha
    sort = arr.argsort()

    
    return colour,sort
