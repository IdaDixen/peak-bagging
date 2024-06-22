#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:19:17 2023

@author: idadixenskaarupwolm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, correlate
from sklearn.linear_model import LinearRegression
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle


# from echelle import plot_echelle # Source: https://github.com/danhey/echelle



def get_target_list():
    return ['Gemma', 'Pooh', 'Boogie', '35 Dra', '36 Dra', 'HD']


def get_target_info(target):
    if target == 'Gemma':
        InputCatalog = 'KIC'
        ID = '11026764'
        name = 'Gemma'
        fmin = 500
        fmax = 1300
        sigma = 30
        s_corr = 250
        s_numax = 4000
        
    elif target == 'Pooh':
        InputCatalog = 'KIC'
        ID = '4351319'
        name = 'Pooh'
        fmin = 240
        fmax = 550
        sigma = 30
        s_corr = 100
        s_numax = 2000
        
    elif target == 'Boogie':
        InputCatalog = 'KIC'
        ID = '11395018'
        name = 'Boogie'
        fmin = 550
        fmax = 1100
        sigma = 50
        s_corr = 135
        s_numax = 2000
        
    elif target == '35 Dra':
        InputCatalog = 'TIC'
        ID = '441813918'
        name = '35 Draconis'
        fmin = 350
        fmax = 1100
        sigma = 50
        s_corr = 100
        s_numax = 5000
        
    elif target == '36 Dra':
        InputCatalog = 'TIC'
        ID = '233121747'
        name = '36 Draconis'
        fmin = 800
        fmax = 1900
        sigma = 50
        s_corr = 100
        s_numax = 7000
        
    elif target == 'HD':
        InputCatalog = 'TIC'
        ID = '459978312'
        name = 'HD 154633'
        fmin = 60
        fmax = 160
        sigma = 10
        s_corr = 20
        s_numax = 1000
    
    else:
        print('Target is not defined')
        return None
    
    return InputCatalog, ID, name, fmin, fmax, sigma, s_corr, s_numax


def get_Dnu(target):
    
    Dnus = np.loadtxt('MyFunctions/Find_peaks/Dnus.csv', delimiter=',', dtype=str)
    Dnu = np.array(Dnus[Dnus[:,0]==target][:,1], dtype='float')
    
    if len(Dnu)==1:
        return Dnu[0]
    
    else:
        print('Large frequency separation is not defined for this target')
        return None


def plotting_colors():
    c0 = 'green'
    c1 = 'mediumblue'
    c2 = 'r'
    c3 = 'magenta'
    
    return c0, c1, c2, c3


def get_markers():
    m0 = 'o'
    m1 = '^'
    m2 = 's'
    m3 = 'h'
    
    return m0, m1, m2, m3


def get_modes(mode_dir=None, target=None, fmin=None, fmax=None, only_radial=False):
    
    if mode_dir is None and target is None:
        print('Either mode_dir or target must be given as input')
        return None
    
    if mode_dir is None:
        InputCatalog, ID, _, _, _, _, _, _ = get_target_info(target)
        mode_dir = 'MyFunctions/Modes/modes_{}{}.csv'.format(InputCatalog, ID)
    
    modes = Table.read(mode_dir)
    
    if fmin is not None:
        modes = modes[np.where((modes['nu']>fmin)&(modes['nu']<fmax))[0]]
    
    if only_radial:
          return modes[modes['l']==0]
    else:
          return modes


def plot_auto_corr(nu, pd, fmin=None, fmax=None, sigma=100, plot_peaks=True):
    
    if fmin is not None:
        mask = np.where((nu>fmin) & (nu<fmax))[0]
        nu = nu[mask]
        pd = pd[mask]
    
    
    smoothed_pd = gaussian_filter1d(pd, sigma=sigma)
    
    pd_to_corr = smoothed_pd-np.median(smoothed_pd)
    corr = correlate(pd_to_corr, pd_to_corr, mode='same')
    # corr = np.ncorrelate(pd_to_corr, pd_to_corr, mode='full')
    # corr = correlate(P_smooth, P_smooth, mode='same')
    
    # f_corr_plot = freqs[int(len(corr)/2):]-freqs[int(len(corr)/2):][0]
    f_corr_plot = nu[int(len(corr)/2):]-nu[int(len(corr)/2):][0]
    corr_plot = corr[int(len(corr)/2):]

    plt.figure()
    plt.plot(f_corr_plot, corr_plot)

    corr_peaks, _ = find_peaks(corr_plot, height=-1)
    
    mask = np.zeros(len(corr_peaks), dtype=bool)
    for i in range(len(mask)): 
        if i%2 == 1: 
            mask[i]=True
    corr_peaks = corr_peaks[mask]
    Dnu_multiples = f_corr_plot[corr_peaks]
    
    if plot_peaks:
        plt.plot(Dnu_multiples, corr_plot[corr_peaks], 'rx', markersize=8)
   
    plt.xlabel(r'$\nu$ [$\mu$Hz]')
    plt.ylabel('Correlation [?]')
    
    # print(corr_peaks)
    # return np.array(f_corr_plot[corr_peaks])
    return corr_peaks, Dnu_multiples, corr_plot[corr_peaks]


def find_numax(nu, pd, fmin, fmax, sigma=2000, plot_peaks=True, ax=None, name=None):
    
    # mean_pd = np.median(pd)
    smoothed_pd = plot_PDS(nu, pd, smooth=True, sigma=sigma, loglog=True)
    
    mask = np.where((nu>fmin) & (nu<fmax))[0]
    nu = nu[mask]
    smoothed_pd = smoothed_pd[mask]
    
    numax_indx = np.argmax(smoothed_pd)
    nu_max = nu[numax_indx]
    pd_max = smoothed_pd[numax_indx]
    
    # fwhm_indx = np.argmin(smoothed_pd-pd_max/2)
    # fwhm_nu = nu[fwhm_indx]
    # fwhm_pd = smoothed_pd[fwhm_indx]
    
    plt.plot(nu_max, pd_max, 'r.')
    # plt.plot(fwhm_nu, fwhm_pd, 'b.')
    plt.legend(loc='lower left')
    
    if ax is not None:
        ax.loglog(nu, smoothed_pd, linewidth=.6)
        ax.text(nu[numax_indx], smoothed_pd[numax_indx], name, fontsize=8)
        ax.set_xlabel(r'$\nu$ [$\mu$Hz]')
        ax.set_ylabel('Power Density [ppm$^2/\mu$Hz]')
        # ax.set_xscale('log')
    
    return nu_max, pd_max#, fwhm_nu
    
    
def find_Dnu(corr_peaks, Dnu_multiples):
    ## plot multiples of Dnu ##
    plt.figure()
    plt.plot(np.arange(len(corr_peaks))+1, Dnu_multiples, '.')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    NDnus = np.arange(len(corr_peaks))+1
    NDnus = NDnus.reshape(len(corr_peaks),1)

    ## linear regression ##
    model = LinearRegression().fit(NDnus, Dnu_multiples)
    model.score(NDnus, Dnu_multiples)
    Dnu = model.coef_
    # print('slope: %f'%Dnu)
    y_predicted = model.predict(NDnus)

    plt.plot(NDnus, y_predicted)
    plt.xlabel(r'Peak number')
    plt.ylabel(r'$\nu$ [$\mu$Hz]')
    # plt.text(3.6,170, r'slope: $\Delta\nu$=%.3f'%Dnu)
    plt.text(0.05, 0.95, r'slope: $\Delta\nu$=%.3f'%Dnu, transform=ax.transAxes)
    
    return Dnu
    
    

def plot_PDS(nu, pd, fmin=None, fmax=None, axs=None,
             smooth=False, sigma=6, loglog=False,
             plot_modes=False, modes=None, mode_dir=None,
             plot_undef=False, fitting=False, wf=False,
             max_peak=False, peak_finder=False, p_height=10):
    
    if wf:
        c_pds = 'k'
        c_smooth = 'k'
        pos_leg1 = [0, 0.8]
    elif fitting:
        c_pds = 'gray'
        c_smooth = 'k'
        pos_leg1 = [0, 0.8]
    elif peak_finder:
        c_pds = 'gray'
        c_smooth = 'k'
        pos_leg1 = [0, 0.1]
    else:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        c_pds = 'gray'
        c_smooth = 'k'#default_colors[1]#'grey'
        pos_leg1 = [0, 0.1]
    
    if fmin is not None:
        mask = np.where((nu>fmin) & (nu<fmax))[0]
        nu = nu[mask]
        pd = pd[mask]
    
    
    if loglog:
        if axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(7,4))
            plt.subplots_adjust(top=0.92,
                                bottom=0.125,
                                left=0.14,
                                right=0.7,
                                hspace=0.5,
                                wspace=0.2)
            
            # fig, axs = plt.subplots(1, 1, figsize=(8,3))
            # plt.subplots_adjust(top=0.92,
            #                     bottom=0.155,
            #                     left=0.1,
            #                     right=0.65,
            #                     hspace=0.5,
            #                     wspace=0.2)
            
        axs.loglog(nu, pd, color=c_pds, linewidth=1.2, label='Weighted PDS')
        axs.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axs.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axs.set_xlim([min(nu)+10,max(nu)])
        # axs.set_xlim([min(nu),max(nu)])
        
    else:
        if axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(10,4))
            plt.subplots_adjust(top=0.92,
                                bottom=0.12,
                                left=0.07,
                                right=0.98,
                                hspace=0.5,
                                wspace=0.2)
            
        axs.plot(nu, pd, color=c_pds, linewidth=1.2, label='Weighted PDS')
        
        axs.set_xlim([min(nu),max(nu)])
        axs.set_ylim([0,max(pd)+30])
        
    axs.set_xlabel(r'$\nu$ [$\mu$Hz]')
    axs.set_ylabel('Power Density [ppm$^2/\mu$Hz]')
    
    if smooth:
        smoothed_pd = gaussian_filter1d(pd, sigma)
        axs.plot(nu, smoothed_pd, color=c_smooth, linewidth=1, 
                  # label='Gaussian kernel \nsmoothed PDS ($\sigma=%i$ $\mu$Hz)'%sigma)
                 label='Smoothed PDS \n($\sigma=%i$ $\mu$Hz)'%sigma)
    
    
    if plot_modes:
        
        if modes is not None:
            l0 = modes[modes['l']==0]['nu']
            l1 = modes[modes['l']==1]['nu']
            l2 = modes[modes['l']==2]['nu']
            l3 = modes[modes['l']==3]['nu']
            
        elif mode_dir is not None:
            l0 = np.unique(np.loadtxt(mode_dir+'_l0.csv'))
            l1 = np.unique(np.loadtxt(mode_dir+'_l1.csv'))
            l2 = np.unique(np.loadtxt(mode_dir+'_l2.csv'))
            l3 = np.unique(np.loadtxt(mode_dir+'_l3.csv'))
            l_undef = np.unique(np.loadtxt(mode_dir+'_undef.csv'))
    
        else:
            print('No modes or directory is specified')
            return None
        
        c0, c1, c2, c3 = plotting_colors()
        m0, m1, m2, m3 = get_markers()
    
        # plot modes found
        # plot_height = max(pd)+70
        # axs.set_ylim([0,plot_height+30])
        # plot_height = max(pd)*1.27
        axs.set_ylim([0,max(pd)*1.27])
        
        if plot_undef:
            axs.scatter(l_undef, np.ones(len(l_undef))*110, s=80, zorder=10,
                        marker='v', color='grey', edgecolors='k', label='Undefined')
        
        axs.scatter(l0, np.ones(len(l0))*(max(pd)*1.18), s=80, zorder=10,
                    marker=m0, color=c0, edgecolors='k', label=r'$\ell=0$', alpha=.8)
        axs.scatter(l1, np.ones(len(l1))*(max(pd)*1.08), s=80, zorder=10,
                    marker=m1, color=c1, edgecolors='k', label=r'$\ell=1$', alpha=.8)
        axs.scatter(l2, np.ones(len(l2))*(max(pd)*.98), s=80, zorder=10,
                    marker=m2, color=c2, edgecolors='k', label=r'$\ell=2$', alpha=.8)
        axs.scatter(l3, np.ones(len(l3))*(max(pd)*.88), s=80, zorder=10,
                    marker=m3, color=c3, edgecolors='k', label=r'$\ell=3$', alpha=.8)
        
        # axs.scatter(l1, np.ones(len(l1))*(50), s=80, zorder=10,
        #             marker=m1, color=c1, edgecolors='k', label=r'$\ell=1$', alpha=.8)
        # axs.scatter(l0, np.ones(len(l0))*(80), s=80, zorder=10,
        #             marker=m0, color=c0, edgecolors='k', label=r'$\ell=0$', alpha=.8)
        # axs.scatter(l2, np.ones(len(l2))*(110), s=80, zorder=10,
        #             marker=m2, color=c2, edgecolors='k', label=r'$\ell=2$', alpha=.8)
        
        if not fitting:
            #save plot handles
            handles, labels = axs.get_legend_handles_labels()
            
            if smooth:# & (not fitting):
                line_handles = handles[0:2]
                mode_handles = handles[2:]
                
            else:
                line_handles = handles[0:1]
                mode_handles = handles[1:]
            
            first_legend = axs.legend(handles=line_handles, fontsize='8',
                                      bbox_to_anchor=pos_leg1, loc='lower left', 
                                      edgecolor=(0, 0, 0, .1))
            # first_legend = axs.legend(handles=line_handles, fontsize='8', loc='upper left')
            # first_legend = axs.legend(handles=line_handles, fontsize='8',
            #                           bbox_to_anchor=[0, 1.1], loc='lower left', 
            #                           edgecolor=(0, 0, 0, .1))
            
            first_legend.get_frame().set_alpha(0.4)
        
            
            # Add the legend manually to the Axes.
            axs.add_artist(first_legend)
            
            second_legend = axs.legend(handles=mode_handles, loc='upper right', 
                                        edgecolor=(0, 0, 0, .1))
            
            second_legend.get_frame().set_alpha(0.4)
        
    else:
        # axs.set_ylim([0,max(pd)+max(pd)*0.05])
        axs.legend(loc='upper right')
    
    
    if max_peak & smooth:
        
        maximum = np.argmax(smoothed_pd)
        
        h = smoothed_pd[maximum]
        pos = nu[maximum]
        
        return [h, pos]
    
    elif peak_finder & smooth:
        
        fit_peaks, _ = find_peaks(smoothed_pd, height=p_height)
        axs.plot(nu[fit_peaks], smoothed_pd[fit_peaks], 'r.', marker=7, markersize=10)
        
        return nu[fit_peaks], smoothed_pd[fit_peaks]
    
    elif smooth and not peak_finder:
        return smoothed_pd
    
    else:
        return None
    

def plot_stacked_ls(nu, pd, ls):
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(7,5))
    plt.subplots_adjust(top=0.99,
                        bottom=0.1,
                        left=0.1,
                        right=0.99,
                        hspace=0.5,
                        wspace=0.3)
    
    pd_all = []
    nu_equal = []
    shift = 0
    for l in ls:
        nu_equal = nu-l
        
        # plot all individual l=1 above on another
        axs[0].plot(nu_equal, pd+shift, 'k', linewidth=0.5)
        shift += 500
        
        # save the corresponding power densities values
        pd_all.append(pd[(nu_equal>-4) & (nu_equal<4)])
        
    # plot l=1 with the summed power
    nu_stacked = nu_equal[(nu_equal>-4) & (nu_equal<4)]
    pd_stacked = sum(pd_all)
    
    axs[1].plot(nu_equal[(nu_equal>-4) & (nu_equal<4)], sum(pd_all), 
                'gray', linewidth=0.5, label='Summed power density')
    
    plt.xlim([-4,4])
    axs[0].set_xlabel(r'Normalized $\nu$ [$\mu$Hz]')
    axs[1].set_xlabel(r'Normalized $\nu$ [$\mu$Hz]')
    axs[0].set_ylabel('Power Density [ppm$^2/\mu$Hz]')
    axs[1].set_ylabel('Power Density [ppm$^2/\mu$Hz]')
    
    # plt.title('l=1 modes for KIC 11026764')
    # build a rectangle in axes coords
    # left, width = .25, .5
    # bottom, height = .25, .5
    # right = left + width
    # top = bottom + height
    # axs[0].text(right, top, r'All individual l=1 \n($\nu$ increases upwards)')
    
    
    smoothed_pd = gaussian_filter1d(pd_stacked, 6)
    axs[1].plot(nu_stacked, smoothed_pd, 'r', linewidth=.8, 
             label='Gaussian kernel \nsmoothed PDS ($\sigma=%i$ $\mu$Hz)'%6)
    axs[1].legend()
    
    maximum = np.argmax(smoothed_pd)
    
    h = smoothed_pd[maximum]
    pos = nu[maximum]
    
    return nu_stacked, pd_stacked, h, pos


# def echelle_diagram(nu, pd, target=None, Dnu=None, cmap='binary', 
#                     fmin=450, fmax=1300, offset=0,
#                     smooth=False, sigma=None,
#                     plot_modes=False, modes=None, mode_dir=None, plot_undef=False,
#                     plot_peaks=False, peaks=None):
    
#     #### plot echelle diagram ####
#     fig, ax = plt.subplots(1, 1, figsize=(7,5))
#     plt.subplots_adjust(top=0.92,
#                         bottom=0.1,
#                         left=0.1,
#                         right=0.99,
#                         hspace=0.5,
#                         wspace=0.2)
    
#     # nu = nu+offset
    
#     if Dnu is None:
#         Dnu = get_Dnu(target)
        
#         if target is None:
#             print('Either target og Dnu need to be defined')
#             return None
    
#     if smooth:
#         if sigma is None:
#             sigma = int((max(nu)-min(nu))/2000)
        
#         plot_echelle(nu, gaussian_filter1d(pd, sigma), Dnu, 
#                      fmin=fmin, fmax=fmax,
#                      scale='sqrt', interpolation='bicubic', cmap=cmap, ax=ax)
#     else:
#         plot_echelle(nu, pd, Dnu, fmin=fmin, fmax=fmax,
#                      scale='sqrt', interpolation='bicubic', cmap=cmap, ax=ax)
    
    
#     plt.ylim([fmin,fmax-Dnu])
#     plt.ylabel(r'$\nu$ [$\mu$Hz]')
#     plt.xlabel(r'$\nu$ mod $\Delta\nu$ [$\mu$Hz]')
    
#     # modes is a nested array with the following ordering of modes [0,1,2,3,undef]
#     if plot_modes:
#         ## overplot found frequencies ##
        
#         if modes is not None:
#             l0 = modes[modes['l']==0]['nu']
#             l1 = modes[modes['l']==1]['nu']
#             l2 = modes[modes['l']==2]['nu']
#             l3 = modes[modes['l']==3]['nu']
            
#         elif mode_dir is not None:
#             l0 = np.unique(np.loadtxt(mode_dir+'_l0.csv'))
#             l1 = np.unique(np.loadtxt(mode_dir+'_l1.csv'))
#             l2 = np.unique(np.loadtxt(mode_dir+'_l2.csv'))
#             l3 = np.unique(np.loadtxt(mode_dir+'_l3.csv'))
#             l_undef = np.unique(np.loadtxt(mode_dir+'_undef.csv'))
    
#         else:
#             print('No modes or directory is specified')
#             return None
        
#         # l0, l1, l2, l3, l_undef = get_modes(mode_dir)
        
#         x_l0 = np.mod(l0, Dnu)#+offset
#         y_l0 = l0
#         x_l1 = np.mod(l1, Dnu)#+offset
#         y_l1 = l1
#         x_l2 = np.mod(l2, Dnu)#+offset
#         y_l2 = l2
#         x_l3 = np.mod(l3, Dnu)#+offset
#         y_l3 = l3
        
#         c0, c1, c2, c3 = plotting_colors()
#         m0, m1, m2, m3 = get_markers()
        
#         # l=0
#         plt.scatter(x_l0, y_l0, s=30, label='$\mathscr{l}=0$',
#                     marker=m0, facecolors=c0, edgecolors='k', alpha=0.4)
#         # l=1
#         plt.scatter(x_l1, y_l1, s=30, label='$\mathscr{l}=1$',
#                     marker=m1, facecolors='None', edgecolors=c1)
#         # l=2
#         plt.scatter(x_l2, y_l2, s=30, label='$\mathscr{l}=2$', 
#                     marker=m2, facecolors='None', edgecolors=c2)
#         # l=3
#         plt.scatter(x_l3, y_l3, s=30, label='$\mathscr{l}=3$',
#                     marker=m3, facecolors='None', edgecolors=c3)
        
#         if plot_undef:
#             x_rest = np.mod(l_undef, Dnu)
#             y_rest = l_undef
#             plt.scatter(x_rest, y_rest, s=8, label='Undefined',
#                         marker='.', color='k')
        
#         plt.legend()
        
#     if plot_peaks:
        
#         plt.scatter(np.mod(peaks, Dnu), peaks, s=8, label='Undefined',
#                     marker='.', color='r')
    
#         plt.legend()
        

def get_TS(file, save_TS=False, to_file=None, plot_TS=True, 
           LS=False, plot_LS=True, 
           smooth=False, sigma=10):
    hdul = fits.open(file)
    data = hdul[1].data
    
    # plt.figure()
    # plt.hist(data['KASOC_FLAG'][data['KASOC_FLAG'] !=0],bins=10)
    
    t = Table(data)
    
    # remove bad data
    t = t[t['KASOC_FLAG']!=1]
    N_all = len(t)
    N_removed = len(t[t['KASOC_FLAG'] ==1])
    N_rest = len(t[t['KASOC_FLAG']!=1])
    if N_removed == 0:
        print('No data points were removed')
    else:
        print('%i data points removed from a total of %i. %i data points remaining.' 
              %(N_removed, N_all, N_rest))
    
    
    # remove nan values
    t = t[~np.isnan(t['FLUX'])]
    
    # get timesteps in days by subtracting first t
    t['TIME']=t['TIME']-t['TIME'][0]
    
    if save_TS:
        t['TIME','FLUX','FLUX_ERR'].write(to_file, format='ascii', overwrite=True)
    
    # Plot time series
    if plot_TS and not LS:
        fig, axs = plt.subplots(1, 1, figsize=(10,4))
        plt.subplots_adjust(top=0.99,
                            bottom=0.09,
                            left=0.075,
                            right=0.985,
                            hspace=0.3,
                            wspace=0.2)
        
        axs.scatter(t['TIME'], t['FLUX'], s=.1, alpha=.6, color='k')
        axs.set_xlabel('Time [days]')# BKJD???
        axs.set_ylabel('Flux [ppm]')
        # axs.set_xlim([t['TIME'][0], t['TIME'][-1]])
        
        return t
    
    # Calculate Lomb-Scargle frequencies and power
    if LS:
        frequency, power = LombScargle(t['TIME'], t['FLUX'],
                                       t['FLUX_ERR']).autopower(nyquist_factor=2)
        
        if plot_LS:
            
            # plt.figure()
            fig, axs = plt.subplots(2, 1, figsize=(10,6))
            plt.subplots_adjust(top=0.99,
                                bottom=0.09,
                                left=0.075,
                                right=0.98,
                                hspace=0.3,
                                wspace=0.2)
            
            # axs[0].scatter(t['TIME'], t['FLUX'], s=.1, alpha=.6, color='gray')
            axs[0].scatter(t['TIME'], t['FLUX'], s=.1, alpha=.6, color='k')
            axs[0].set_xlabel('Time [days]')# BKJD???
            axs[0].set_ylabel('Normalized flux [ppm]')
            # axs[0].set_xlim([t['TIME'][0], t['TIME'][-1]])
            
            
            plot_LS_ps(frequency.value, power.value, ax=axs[1],
                       smooth=smooth, sigma=sigma)
        
        return t, frequency.value, power.value


def plot_LS_ps(frequency, power, fmin=500, fmax=1300, 
               ax=None, loglog=False,
               smooth=False, sigma=10):
    
    # take care of units
    freqs = frequency/(24*60*60)*1000000 # conversion: 1/d --> 1/s and to muHz
    mask = np.where((freqs>fmin) & (freqs<fmax))[0]
    Ps = power*1000000 # conversion to ppm
    
    # Plot powerspectrum
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,4))
        plt.subplots_adjust(top=0.99,
                            bottom=0.115,
                            left=0.065,
                            right=0.995,
                            hspace=0.3,
                            wspace=0.2)
        
        if loglog:
            ax.loglog(freqs, Ps, color='k', linewidth=.3,
                      label='Lomb-Scargle Periodogram')
            ax.set_xlabel(r'$\nu$ [$\mu$Hz]')# BKJD???
            ax.set_ylabel(r'Power [$ppm^2$/$\mu$Hz]')
            
            # plt.savefig('psKIC11026764.png', dpi=300)
            
        else:
            ax.plot(freqs[mask], Ps[mask], color='k', linewidth=.3,
                    label='Lomb-Scargle Periodogram')
            ax.set_xlabel(r'$\nu$ [$\mu$Hz]')# BKJD???
            ax.set_ylabel(r'Power [$ppm^2$/$\mu$Hz]')
        
            # plt.savefig('psKIC11026764.png', dpi=300)
            
    else:
        if loglog:
            ax.loglog(freqs, Ps, color='k', linewidth=.3,
                      label='Lomb-Scargle Periodogram')
            ax.set_xlabel(r'$\nu$ [$\mu$Hz]')
            ax.set_ylabel('Power [$ppm^2$/$\mu$Hz]')
            
            # plt.savefig('psKIC11026764.png', dpi=300)
        else:
            ax.plot(freqs[mask], Ps[mask], color='k', linewidth=.3,
                    label='Lomb-Scargle Periodogram')
            ax.set_xlabel('Frequency [$\mu$Hz]')
            ax.set_ylabel('Power [$ppm^2$/$\mu$Hz]')
        
    if smooth and not loglog:
        smoothed_pd = gaussian_filter1d(Ps[mask], sigma)
        ax.plot(freqs[mask], smoothed_pd, 'r', linewidth=.8, 
                label='Gaussian kernel \nsmoothed PDS ($\sigma=%i$ $\mu$Hz)'%sigma)
    
    plt.legend()











