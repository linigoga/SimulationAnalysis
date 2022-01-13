"""
This is a simple analysis file which garners all the EPOCH
.sdf files. It then generates figures and calculates the
instablity growth rate.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import sdf
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm



def calc_fft(x,signal):
    signal = gaussian_filter(signal/np.sum(signal),1)
    dx = x[1]-x[0]
    freq = 2*np.pi*np.fft.fftfreq(len(signal), dx)
    fmax = int(len(freq)/2)
    freq = freq[1:fmax]
    fft = np.abs(np.fft.fft(signal))[1:fmax]
    
    return freq,fft


def derivative(a,b,h):
    return (b - a)/h


def make_dir(loc):
    location = os.path.join(loc,'figures')
    try:
        os.makedirs(location)
    except OSError:
        if not os.path.join(location):
            raise

    return location


def read_file(file):
    data = sdf.read(file, derived=True,dict=True)
    n1 = data['Derived/Number_Density/bunch1'].data
    n2 = data['Derived/Number_Density/bunch2'].data
    time = data['Header']['time']
    x = data['Grid/Grid'].data[0][:-1]
    y = data['Grid/Grid'].data[1][:-1]

    cache = [n1,n2,time,x,y]

    return cache

def plot(cache,location):


    fontsize = 18
    n1,n1_y,n2,n2_y,time,x,y,freq1,fft1,freq2,fft2,file = cache

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(221)
    im = ax.imshow(np.rot90(n1),
                   extent=1e6*np.array([-x.max(),x.max(),-x.max(),x.max()])
                   ,aspect='auto')
    cbar1 = fig.colorbar(im,ax=ax,pad=0)
    cbar1.set_label('$n_e, m^{-3}$',fontsize = fontsize)
    
    
    ax.set_xlabel('$x, \mu m$',fontsize = fontsize)
    ax.set_xlabel('$y, \mu m$',fontsize = fontsize)
    ax.tick_params(labelsize = fontsize)

    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(np.rot90(n2),
                   extent=1e6*np.array([-x.max(),x.max(),-x.max(),x.max()])
                   ,aspect='auto')
    cbar2 = fig.colorbar(im2,ax=ax2,pad=0)
    cbar2.set_label('$n_e, m^{-3}$',fontsize = fontsize)
    
    
    ax2.set_xlabel('$x, \mu m$',fontsize = fontsize)
    ax2.set_xlabel('$y, \mu m$',fontsize = fontsize)
    ax2.tick_params(labelsize = fontsize)

    #signals
    
    ax3 = fig.add_subplot(223)
    
    ax3.plot(1e6*np.array(y),n1_y,alpha = 0.8,label='slab 1')
    ax3.plot(1e6*np.array(y),n2_y,alpha = 0.8,label='slab 2')
    
    ax3.set_ylabel('$n_e(y), m^{-3} $',fontsize = fontsize)
    ax3.set_xlabel('$y, \mu m$',fontsize = fontsize)
    ax3.legend(loc='best')
    ax3.tick_params(labelsize=fontsize)

    # FFT
    ax4 = fig.add_subplot(224)
    
    ax4.plot(freq1,fft1[:249],alpha = 0.8,label='slab 1')
    ax4.plot(freq2,fft2[:249],alpha = 0.8,label='slab 2')
    
    ax4.set_xlabel('k_y, $m^{-1}$',fontsize = fontsize)
    ax4.set_ylabel('spectral intensity \n arb. units',fontsize = fontsize)
    ax4.tick_params(labelsize = fontsize)
    
    plt.suptitle(f'{np.round(time,2)*1e15}',y=1.01)
    plt.tight_layout()

    fname = file.split('/')[-1].split('.sdf')[0]


    plt.savefig(f'{os.path.join(location,fname)}.png',dpi = 300)
    
    plt.clf()
    plt.close()
    return



def analysis(file,location):
    n1,n2,time,x,y = read_file(file)
    
    n1_y = np.average(n1,axis=1)
    n2_y = np.average(n2,axis=1)
    
    freq1,fft1 = calc_fft(y,n1_y)
    freq2,fft2 = calc_fft(y,n2_y)

    cache = [n1,n1_y,n2,n2_y,time,x,y,freq1,fft1,freq2,fft2,file]



    plot(cache,location)


    return fft1,fft2,time


def plot_ffts(fft1,fft2,sigs,times,location):
    
    fontsize = 18
    fig = plt.figure(figsize=(6,8))

    ax = fig.add_subplot(121)

    ax.plot(times,fft1,label='$n_1$, fft',c='r')
    ax.plot(times,fft2,label='$n_2$, fft',c='b')
    ax.plot(times,sigs,label='fft, both',c='orange')

    ax.set_xlabel("Time, (fs)",fontsize = fontsize)
    ax.set_ylabel("Spectral amplitude \n arb. units")
    ax.legend(loc = 'best')
    ax.tick_params(labelsize=fontsize)


    ax2 = fig.add_subplot(122)

    dt = times[2] - times[1]
    gr_1 = [derivative(fft1[i],fft1[i+1],dt) for i in range(len(fft1[:-1]))]
    gr_2 = [derivative(fft2[i],fft2[i+1],dt) for i in range(len(fft2[:-1]))]
    gr_s = [derivative(sigs[i],sigs[i+1],dt) for i in range(len(sigs[:-1]))]

    ax2.plot(times[:-1],gr_1,label='Slab 1',c='r')
    ax2.plot(times[:-1],gr_2,label='$Slab 2',c='b')
    ax2.plot(times[:-1],gr_s,label='Both',c='orange')

    ax2.set_xlabel("Time, (fs)",fontsize = fontsize)
    ax2.set_ylabel("growth rate, $s^{-1}$",fontsize = fontsize)
    ax2.legend(loc = 'best')
    ax2.tick_params(labelsize=fontsize)
    fname = os.path.join(location,'growths.png')

    plt.tight_layout()
    plt.savefig(fname,dpi=300)
    plt.clf()
    plt.close()


    return



def main():
    
    directories = list(glob.glob('ppc_*'))
    try:
        inp = int(input(f"choose {directories} "))
    except EOFError:
        pass
    
    print(directories)


    if inp <= len(directories):
        print(f"you've chosen {directories[inp]}")
    else:
        print('invalid request. terminating')
        return

    loc = directories[inp]
    
    files = glob.glob(f'{loc}/*.sdf')
    files.sort()

    fft_1s,fft_2s = [],[]
    times,all_sigs = [],[]
    for file in tqdm(files[:-3]):
        location = make_dir(loc)
        fft1,fft2,time = analysis(file,location)
        
        times.append(time*1e15)
        fft_1s.append(np.max(fft1))
        fft_2s.append(np.max(fft2))

        all_sigs.append(np.max(fft1) + np.max(fft2))

    plot_ffts(fft_1s,fft_2s,all_sigs,times,location)


    return


if __name__ == "__main__":
    main()