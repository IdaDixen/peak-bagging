#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:22:18 2023

@author: Ida Dixen Skaarup Wölm


###############################################################################

                     Oscillation modes modelling with MCMC                     

###############################################################################


"""


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, ScalarFormatter, MaxNLocator
from scipy.special import lpmv

import string
from multiprocessing import Pool

import emcee

import functions as myF
import MCMC_bg as bg
from priors import uniform_prior, Jeffreys_prior, mod_Jeffreys_prior, truncated_Gaussian_prior
from results_plotting import PlotBestFit



"""
###############################################################################

                               Parameter control                               

###############################################################################
"""

def get_labels(modes, fmin=None, fmax=None, 
               only_radial=False, #F_like=False,
               fit_split=True, mixed_modes=True,
               fit_bg=True, bg_k=2):
    
    nu_nl = []
    A_nl = []
    Gamma_nl = []
    nu_sn = []
    V_n1 = []

    # count numbers of l=1 modes
    n_groups = modes.group_by('n')
    l1_add = string.ascii_lowercase
    multiple_l1 = []
    for group in n_groups.groups:
        if len(group[group['l']==1]) > 1:
            multiple_l1.append(True)
        else:
            multiple_l1.append(False)
    
    # make labels to every mode
    j = 0
    k = 0
    for i, mode in enumerate(modes):
        n = mode['n']
        l = mode['l']
        
        # keep track of the n value
        if i == n_groups.groups.indices[j+1]:
            j += 1
            k = 0
        
        # index mixed modes
        # append letter if more than one l=1 value in that order
        if (l==1) & (multiple_l1[j]):
            l_str = str(l)+l1_add[k]
            k += 1
        else:
            l_str = str(l)
        
        # make labels
        if only_radial:
            if l == 0:
                nu_nl.append(r'$\nu_{'+str(n)+','+l_str+'}$')
                A_nl.append(r'$A_{'+str(n)+','+l_str+'}$')
                Gamma_nl.append('$\Gamma_{'+str(n)+','+l_str+'}$')
            
            fit_split = False
            mixed_modes = False
            
        else:
            nu_nl.append(r'$\nu_{'+str(n)+','+l_str+'}$')
            
            if l == 0:
                A_nl.append(r'$A_{'+str(n)+','+l_str+'}$')
                Gamma_nl.append(r'$\Gamma_{'+str(n)+','+l_str+'}$')
                
            if (l == 1) & (mixed_modes):
                A_nl.append(r'$A_{'+str(n)+','+l_str+'}$')
                V_n1.append(r'$V_{'+str(n)+','+l_str+'}$')
                Gamma_nl.append(r'$\Gamma_{'+str(n)+','+l_str+'}$')
                nu_sn.append(r'$\nu_{s,'+str(n)+','+l_str+'}$')
                
    
    labels = np.concatenate((nu_nl, A_nl, Gamma_nl))

    if not only_radial:
        
        ls = np.unique(modes['l'])
        
        if fit_split | mixed_modes:
            l_maxfit = 1
        else:
            l_maxfit = 0
        
        if mixed_modes:
            # add nuiscent visibility per order
            labels = np.concatenate((labels, V_n1))
            
            # add visibility for interpolated modes
            for l in ls[ls>l_maxfit]:
                labels = np.append(labels, r'$V^2_{}$'.format(l))
                # V is refering to V^tilde_l = V_l/V_0
            
            if fit_split:
                # add nuiscent split per order
                labels = np.concatenate((labels, nu_sn, ['i']))
            
        else:
            # add visibility for interpolated modes
            for l in ls[ls>l_maxfit]:
                labels = np.append(labels, r'$V^2_{}$'.format(l))
            
            if fit_split:
                labels = np.concatenate((labels, [r'$\nu_s$', 'i']))
            
    if fit_bg:
        labels = np.concatenate((labels, bg.get_labels(bg_k, gaussian_env=False))) 

    return labels



def get_index(index_type, modes, fit_split=True, mixed_modes=True):
    
    if fit_split | mixed_modes:
        l_maxfit = 1
        mask_fit = (modes['l']==0) | (modes['l']==1)
    else:
        l_maxfit = 0
        mask_fit = (modes['l']==0)
    
    Nmodes = len(modes)
    mask1 = (modes['l']==1)
    Nl1 = len(modes[mask1])
    Nl_fit = len(modes[mask_fit])
    
    A_min = Nmodes
    A_max = A_min+Nl_fit
    Gamma_min = A_max
    Gamma_max = A_max+Nl_fit
    
    if index_type == 'fit':
        index = A_min, A_max, Gamma_min, Gamma_max
    
    ls = np.unique(modes['l'])
    N_Vl = len(ls[ls>l_maxfit])

    if mixed_modes:
        V_nl_min = Gamma_max
        V_nl_max = V_nl_min+Nl1
        
        V_l_min = V_nl_max
        V_l_max = V_l_min+N_Vl
        
        if index_type == 'interpolate':
            index = V_nl_min, V_nl_max, V_l_min, V_l_max
        
        if fit_split:
            #nu_s, i 
            nu_s_min = V_l_max
            nu_s_max = nu_s_min+Nl1
            i_index = nu_s_max
            
            if index_type == 'split':
                index = nu_s_min, nu_s_max, i_index
        
    else:
        V_l_min = Gamma_max
        V_l_max = V_l_min+N_Vl
        
        if index_type == 'interpolate':
            index = V_l_min, V_l_max
        
        if fit_split:
            #nu_s, i 
            nu_s_index = V_l_max
            i_index = V_l_max+1
            
            if index_type == 'split':
                index = nu_s_index, i_index
    
    return index


def interpolate_modes(modes, nu_nl, A_nl, Gamma_nl, l_min, plot=False):#, res_dir=None):
    # interpolate A and Gamma to l>l_min
    # int: interpolated
    # fit: fitted
    
    A_fit = A_nl
    Gamma_fit = Gamma_nl
    
    if l_min == 1:
        mask_fit = (modes['l']==0)
        mask_int = (modes['l']==1) | (modes['l']==2) | (modes['l']==3)
        
    if l_min == 2:
        mask_fit = (modes['l']==0) | (modes['l']==1)
        mask_int = (modes['l']==2) | (modes['l']==3)
    
    mask0 = modes[mask_fit]['l']==0
    
    nu_int = nu_nl[mask_int]
    nu_fit = nu_nl[mask_fit]
    nu0 = nu_fit[mask0]
    A0 = A_nl[mask0]
    Gamma0 = Gamma_nl[mask0]
    
    A_int = np.interp(nu_int, nu0, A0)
    Gamma_int = np.interp(nu_int, nu0, Gamma0)
    
    # merge arrays and sort by nu
    nu_nl = np.append(nu_fit, nu_int)
    sort = np.argsort(nu_nl)
    nu_nl = nu_nl[sort]
    A_nl = np.append(A_nl, A_int)[sort]
    Gamma_nl = np.append(Gamma_nl, Gamma_int)[sort]
    
    if plot:
        # merge arrays and sort by nu
        nu_plot = np.append(nu0, nu_int)
        sort = np.argsort(nu_plot)
        nu_plot = nu_plot[sort]
        A_plot = np.append(A0, A_int)[sort]
        Gamma_plot = np.append(Gamma0, Gamma_int)[sort]
        
        if l_min == 2:
            mask1 = modes[mask_fit]['l']==1
            nu1 = nu_fit[mask1]
            A1 = A_fit[mask1]
            Gamma1 = Gamma_fit[mask1]
        
        # plot A and Gamma init
        fig_int, ax1 = plt.subplots(1, 1, figsize=(7,4))
        ax2 = ax1.twinx()
        plt.subplots_adjust(top=0.92,
                            bottom=0.125,
                            left=0.1,
                            right=0.8,
                            hspace=0.5,
                            wspace=0.2)

        if l_min == 2:
            ax1.plot(nu1, A1, '^', color='gray', label=r'Fitted $\ell=1$')
        lns1 = ax1.plot(nu_plot, A_plot, '.-', linewidth=.3,
                        label=r'$A_{nl}$')
        ax1.plot(nu0, A0, 'k.', label=r'Fitted $\ell=0$')
        ax1.set_xlabel(r'$\nu$ [$\mu$Hz]')
        ax1.set_ylabel(r'$A$ [ppm]')
        
        if l_min == 2:
            lns4 = ax2.plot(nu1, Gamma1, '^', color='gray', label=r'Fitted $\ell=1$')
            
        lns2 = ax2.semilogy(nu_plot, Gamma_plot, 'r.-', linewidth=.3,
                            label=r'$\Gamma_{nl}$')
        lns3 = ax2.semilogy(nu0, Gamma0, 'k.', 
                            label=r'Fitted $\ell=0$')
        lns = lns1+lns2+lns3
        
        if l_min == 2:
            lns = lns+lns4
        ax2.set_ylabel(r'$\Gamma$ [$\mu$Hz]')
        
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')

        # fig_int, ax1 = plt.subplots(1, 1, figsize=(7,4))
        # ax2 = ax1.twinx()
        # plt.subplots_adjust(top=0.92,
        #                     bottom=0.125,
        #                     left=0.1,
        #                     right=0.8,
        #                     hspace=0.5,
        #                     wspace=0.2)
        # 
        # lns1 = ax1.plot(nu_nl, A_nl, '.-', linewidth=.3, 
        #                 label=r'$A_{nl}$')
        # lns3 = ax1.plot(nu_fit, A_fit, 'k*', label=r'$fitted$')
        # ax1.set_xlabel(r'$\nu$ [$\mu$Hz]')
        # ax1.set_ylabel(r'$A$ [ppm]')

        # lns2 = ax2.semilogy(nu_nl, Gamma_nl, 'r.-', linewidth=.3,
        #                     label=r'$\Gamma_{nl}$')
        # 
        # ax2.semilogy(nu_fit, Gamma_fit, 'k*', label=r'$Fitted$')
        # ax2.set_ylabel(r'$\Gamma$ [$\mu$Hz]')
        # #ax2.yaxis.set_minor_formatter(ScalarFormatter())

        # lns = lns1+lns2+lns3
        # labs = [l.get_label() for l in lns]
        # ax1.legend(lns, labs, loc='upper left')
        # 
        # # import matplotlib.ticker as ticker

        # # ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        # 
        # # formatter = ScalarFormatter()
        # # formatter.set_scientific(False)
        # # ax2.set_major_formatter(formatter)
        # # plt.show()
        # 
        # #plt.close(fig_int)
    
    return A_nl, Gamma_nl


def unpack_theta(theta, modes, only_radial=False, 
                 fit_split=True, mixed_modes=True, prior=False):
    
    unpacked = ()
    
    if fit_split | mixed_modes:
        l_min = 2
    else:
        l_min = 1
    
    # mode frequencies
    nu_nl = theta[:len(modes)]
    
    # heights and widths
    A_min, A_max, Gamma_min, Gamma_max = get_index('fit', modes,  
                                                   fit_split=fit_split, 
                                                   mixed_modes=mixed_modes)
    A_nl     = theta[A_min:A_max]
    Gamma_nl = theta[Gamma_min:Gamma_max]
    
    if only_radial & l_min==1:
        unpacked = unpacked + (nu_nl, A_nl, Gamma_nl)
        
    if not only_radial:
        if not prior:
            A_nl, Gamma_nl = interpolate_modes(modes, nu_nl, A_nl, Gamma_nl, l_min)
        
        unpacked = unpacked + (nu_nl, A_nl, Gamma_nl)
        
        # V_l_min, V_l_max, nu_s_index, i_index = get_index('multipole', modes)
        if mixed_modes:
            V_nl_min, V_nl_max, V_l_min, V_l_max = get_index('interpolate', 
                                                             modes,  
                                                             fit_split=fit_split, 
                                                             mixed_modes=mixed_modes)
            
            # visibility incl. nuiscent param due to coupling to g-modes
            V_nl = theta[V_nl_min:V_nl_max]
            
            # geometric visibility for interpolated modes
            if (V_l_min-V_l_max) == 0:
                V_l = theta[V_l_min]
            else:
                V_l = theta[V_l_min:V_l_max]
            unpacked = unpacked + (V_nl, V_l)
            
            if fit_split:
                nu_s_min, nu_s_max, i_index = get_index('split', modes,  
                                                        fit_split=fit_split, 
                                                        mixed_modes=mixed_modes)
                nu_s         = theta[nu_s_min:nu_s_max]
                i            = theta[i_index]
                unpacked = unpacked + (nu_s, i)
        
        else: # if no mixed modes
            V_l_min, V_l_max = get_index('interpolate', modes,  
                                        fit_split=fit_split, 
                                        mixed_modes=mixed_modes)
            if (V_l_min-V_l_max) == 0:
                V_l = theta[V_l_min]
            else:
                V_l = theta[V_l_min:V_l_max]
            unpacked = unpacked + (V_l,)
            
            if fit_split:
                nu_s_index, i_index = get_index('split', modes,  
                                                fit_split=fit_split, 
                                                mixed_modes=mixed_modes)
                nu_s         = theta[nu_s_index]
                i            = theta[i_index]
                unpacked = unpacked + (nu_s, i)
    
    return unpacked



##########  .  ##########



"""
###############################################################################

                         Bayesian inference modelling                         

###############################################################################
"""

##########  Models  ##########



def model(theta, nu, InputCatalog, modes, bg_k, mcmc_sampling=True, 
          only_radial=False, fit_split=True, mixed_modes=True,
          fit_bg=True, BG_bestFit=None):
    
    if mixed_modes:
        if fit_split:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_nl_all, V_l, nu_ns_all, i = unpack_theta(theta, modes, 
                                                                                          only_radial=only_radial,
                                                                                          fit_split=fit_split, 
                                                                                          mixed_modes=mixed_modes)
        else:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_nl_all, V_l = unpack_theta(theta, modes, 
                                                                            only_radial=only_radial,
                                                                            fit_split=fit_split, 
                                                                            mixed_modes=mixed_modes)
    else: 
        if fit_split:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_l, nu_s, i = unpack_theta(theta, modes, 
                                                                           only_radial=only_radial,
                                                                           fit_split=fit_split, 
                                                                           mixed_modes=mixed_modes)
        else:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_l = unpack_theta(theta, modes, 
                                                                  only_radial=only_radial,
                                                                  fit_split=fit_split, 
                                                                  mixed_modes=mixed_modes)
        
    
    
    if fit_bg:
        sigma_k, tau_k, W = bg.unpack_theta(theta, bg_k, gaussian_env=False)
    else:
        sigma_k, tau_k, W = bg.unpack_theta(BG_bestFit, bg_k, gaussian_env=False)
    
    
    ####  Apodization  ####
    if InputCatalog == 'KIC':
        Delta_t = 58.89
    elif InputCatalog == 'TIC':
        Delta_t = 120
    
    x = np.pi * nu/1e6 * Delta_t
    eta = np.sin(x)/x
    
    
    ####  BACKGROUND  ####
    B = bg.bg_model(nu, sigma_k, tau_k)
    
    
    ####  MODES  ####
    
    if fit_split:
        # fold i into the interval if above 1
        if i > 1:
            i = 1-(i-1)
        #convert from cos(i) to i (in radians)
        i = np.arccos(i)
        
    
    if fit_split | mixed_modes:
        l_min = 2
    else:
        l_min = 1
    
    P_nlm = 0
    
    l1_indx = -1
    for ii, (mode, nu_nl, A_nl, Gamma_nl) in enumerate(zip(modes, 
                                                           nu_nl_all, 
                                                           A_nl_all, 
                                                           Gamma_nl_all)):
        
        # fold A_nl into the interval if below 0
        A_nl = abs(A_nl)

        l = mode['l']
        if l == 1:
            l1_indx += 1
        
        if mixed_modes: # assuming only dipole modes are significantly mixed or splitted
            
            if fit_split:
                # fold nu_s into the interval if below 0
                nu_ns = abs(nu_ns_all[l1_indx])
            
            if l == 0:
                V_nl = 1
            elif l == 1:
                V_nl = V_nl_all[l1_indx]
            else:
                V_nl = V_l[l-l_min]
        
        else:
            if fit_split:
                # fold nu_s into the interval if below 0
                nu_ns = abs(nu_s)
            
            if l == 0:
                V_nl = 1
            else:
                V_nl = V_l[l-l_min]
        
        P_m = 0
        for m in np.arange(-l,l+1):
            if fit_split:
                Eps_lm = math.factorial(l-abs(m))/math.factorial(l+abs(m)) * lpmv(abs(m),l,np.cos(i))**2
            else:
                Eps_lm = 1
                #m = 0
                nu_ns = 0
            
            S_nl = 2 * A_nl**2 / (np.pi * Gamma_nl)
            H_nlm = Eps_lm * V_nl * S_nl
            if not fit_split and m==0:
                P_m += H_nlm / (1 + 4/(Gamma_nl**2) * (nu-nu_nl-m*nu_ns)**2)
            elif fit_split:
                P_m += H_nlm / (1 + 4/(Gamma_nl**2) * (nu-nu_nl-m*nu_ns)**2)            
        
        P_nlm += P_m
    
    return eta**2 * (P_nlm + B) + W

##########  .  ##########



##########  Likelihood function  ##########

def log_likelihood(theta, nu, power, InputCatalog, modes, bg_k, 
                   only_radial=False, fit_split=True, mixed_modes=True,
                   fit_bg=True, BG_bestFit=None):
    
    model_fit = model(theta, nu, InputCatalog, modes, bg_k, 
                      only_radial=only_radial, fit_split=fit_split, mixed_modes=mixed_modes,
                      fit_bg=fit_bg, BG_bestFit=BG_bestFit)
    
    return -np.nansum(np.log(model_fit) + power/model_fit)

##########  .  ##########




##########  Priors  ##########

def log_prior(theta, nu, pd, target, modes, theta_init, bg_k, 
              only_radial=False, fit_split=True, mixed_modes=True,
              fit_bg=True):
    
    V_nl_all = None
    nu_ns_all = None
    
    # unpack trial theta
    if mixed_modes:
        if fit_split:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_nl_all, V_l, nu_ns_all, i = unpack_theta(theta, modes, 
                                                                                          only_radial=only_radial,
                                                                                          fit_split=fit_split, 
                                                                                          mixed_modes=mixed_modes,
                                                                                          prior=True)
        else:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_nl_all, V_l = unpack_theta(theta, modes, 
                                                                            only_radial=only_radial,
                                                                            fit_split=fit_split, 
                                                                            mixed_modes=mixed_modes,
                                                                            prior=True)
    else: 
        if fit_split:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_l, nu_s, i = unpack_theta(theta, modes, 
                                                                           only_radial=only_radial,
                                                                           fit_split=fit_split, 
                                                                           mixed_modes=mixed_modes,
                                                                           prior=True)
        else:
            nu_nl_all, A_nl_all, Gamma_nl_all, V_l = unpack_theta(theta, modes, 
                                                                  only_radial=only_radial,
                                                                  fit_split=fit_split, 
                                                                  mixed_modes=mixed_modes,
                                                                  prior=True)
    
    # unpack initial theta
    if mixed_modes:
        if fit_split:
            nu_nl_inits, A_nl_inits, Gamma_nl_inits, V_nl_inits, V_l_inits, nu_ns_inits, i_inis = unpack_theta(theta_init, modes, 
                                                                                                               only_radial=only_radial,
                                                                                                               fit_split=fit_split, 
                                                                                                               mixed_modes=mixed_modes,
                                                                                                               prior=True)
        else:
            nu_nl_inits, A_nl_inits, Gamma_nl_inits, V_nl_inits, V_l_inits = unpack_theta(theta_init, modes, 
                                                                                          only_radial=only_radial,
                                                                                          fit_split=fit_split, 
                                                                                          mixed_modes=mixed_modes,
                                                                                          prior=True)
    else: 
        if fit_split:
            nu_nl_inits, A_nl_inits, Gamma_nl_inits, V_l_inits, nu_s_init, i_init = unpack_theta(theta_init, modes, 
                                                                                                 only_radial=only_radial,
                                                                                                 fit_split=fit_split, 
                                                                                                 mixed_modes=mixed_modes,
                                                                                                 prior=True)
        else:
            nu_nl_inits, A_nl_inits, Gamma_nl_inits, V_l_inits = unpack_theta(theta_init, modes, 
                                                                              only_radial=only_radial,
                                                                              fit_split=fit_split, 
                                                                              mixed_modes=mixed_modes,
                                                                              prior=True)
    
    
    # unpack background theta (trial and initial)
    if fit_bg:
        sigma_k, tau_k, W = bg.unpack_theta(theta, bg_k, gaussian_env=False)
        sigma_k_init, tau_k_init, W_init = bg.unpack_theta(theta_init, bg_k, gaussian_env=False)
    
    
    InputCatalog, ID, name, fmin, fmax, sigma, s_corr, s_numax = myF.get_target_info(target)
    labels = get_labels(modes, only_radial=only_radial, 
                        fit_split=fit_split, mixed_modes=mixed_modes,
                        fit_bg=fit_bg, bg_k=bg_k)
    
    Dnu = myF.get_Dnu(target)
    
    # add since they are log!
    indx = -1
    for nu_nl, nu_init in zip(nu_nl_all, nu_nl_inits):
        indx +=1
        # lp = uniform_prior(nu_nl, nu_init-4, nu_init+4)
        lp = uniform_prior(nu_nl, nu_init-0.2*Dnu, nu_init+0.2*Dnu)
        if abs(lp) == np.inf:
            print('{} is inf'.format(labels[indx]))
            print('in nu')
            print(nu_nl)
            return lp, indx
        
    
    for A_nl, A_init in zip(A_nl_all, A_nl_inits):        
        indx += 1
        # lp += mod_Jeffreys_prior(A_nl, 0.001, 100, A_init)
        #lp += mod_Jeffreys_prior(A_nl, 0, np.sqrt(max(pd)), A_init)
        lp += mod_Jeffreys_prior(A_nl**2, 0, max(pd), A_init**2)
        if abs(lp) == np.inf:
            print('{} is inf'.format(labels[indx]))
            print('in A')
            print(A_nl)
            return lp, indx
    
    for Gamma_nl, Gamma_init in zip(Gamma_nl_all, Gamma_nl_inits):
        indx += 1
        # lp += mod_Jeffreys_prior(Gamma_nl, 0.001, 10, Gamma_init)
        lp += mod_Jeffreys_prior(Gamma_nl, 0, 20, Gamma_init)
        if abs(lp) == np.inf:
            print('in gamma')
            print('{} is inf'.format(labels[indx]))
            print(Gamma_nl)
            return lp, indx
    
    
    
    if not only_radial:
        
        if fit_split | mixed_modes:
            l_maxfit = 1
        else:
            l_maxfit = 0
        
        V_0s = [1.22, 0.71, 0.14]
        V_maxs = [3, 1, 0.5]
        sigma_gauss = [1.5, 0.5, 0.05]
    
        if mixed_modes:
            # V_n1 prior
            for V_nl, V_nl_init in zip(V_nl_all, V_nl_inits):
                indx += 1
                lp += uniform_prior(V_nl, 0, 5)
                if abs(lp) == np.inf:
                    # print('in V_nl')
                    print('{} is inf'.format(labels[indx]))
                    print(V_nl)
                    return lp, indx
            
            # V_l prior
            for ii, (V, V_l_init) in enumerate(zip(V_l, V_l_inits)):
                indx += 1
                ii += l_maxfit
                V0 = V_0s[ii]
                Vs = sigma_gauss[ii]
                Vmax = V_maxs[ii]
                lp += truncated_Gaussian_prior(V, 0, Vmax, V0, Vs)
                if abs(lp) == np.inf:
                    # print('in V_l')
                    print('{} is inf'.format(labels[indx]))
                    print(V)
                    return lp, indx
            
            if fit_split:
                # nu_ns prior
                for nu_ns, nu_ns_init in zip(nu_ns_all, nu_ns_inits):
                    indx += 1
                    lp += uniform_prior(nu_ns, -1, 3)
                    if abs(lp) == np.inf:
                        # print('in nu_s')
                        print('{} is inf'.format(labels[indx]))
                        print(nu_ns)
                        return lp, indx
                
                #inclination
                indx += 1
                # lp += uniform_prior(i, -180, 90)
                # lp += uniform_prior(i, -np.pi/2, np.pi)
                lp += uniform_prior(i, 0, 1.5)
                if abs(lp) == np.inf:
                    # print('in i')
                    print('{} is inf'.format(labels[indx]))
                    print(i)
                    return lp, indx
        else:
            # V_l prior
            for ii, (V, V_l_init) in enumerate(zip(V_l, V_l_inits)):
                indx += 1
                ii += l_maxfit
                V0 = V_0s[ii]
                Vs = sigma_gauss[ii]
                Vmax = V_maxs[ii]
                lp += truncated_Gaussian_prior(V, 0, Vmax, V0, Vs)
                if abs(lp) == np.inf:
                    print('in V_l')
                    print('{} is inf'.format(labels[indx]))
                    print(V)
                    return lp, indx
            
            if fit_split:
                # nu_s prior
                indx += 1
                lp += uniform_prior(nu_s, -1, 3)
                if abs(lp) == np.inf:
                    print('{} is inf'.format(labels[indx]))
                    print(nu_s)
                    return lp, indx
                
                #inclination
                indx += 1
                # lp += uniform_prior(i, -180, 90)
                # lp += uniform_prior(i, -np.pi/2, np.pi)
                lp += uniform_prior(i, 0, 1.5)
                if abs(lp) == np.inf:
                    print('{} is inf'.format(labels[indx]))
                    print(i)
                    return lp, indx
    
    if fit_bg:
        for sigma, sigma_init in zip(sigma_k, sigma_k_init):
            indx += 1
            sigma0 = sigma_init
            sigma_s = 0.1*sigma0
            lp += truncated_Gaussian_prior(sigma, 0.1*sigma0, 10*sigma0, sigma0, sigma_s)
            if abs(lp) == np.inf:
                print('{} is inf'.format(labels[indx]))
                print(sigma)
                return lp, indx
        
        for tau, tau_init in zip(tau_k, tau_k_init):
            indx += 1
            tau0 = tau_init
            tau_s = 0.1*tau0
            lp += truncated_Gaussian_prior(tau, 0.1*tau0, 10*tau0, tau0, tau_s)
            if abs(lp) == np.inf:
                print('{} is inf'.format(labels[indx]))
                print(sigma)
                return lp, indx
        
        indx += 1
        W0 = W_init
        W_s = 0.1*W0
        lp += truncated_Gaussian_prior(W, 0.1*W0, 10*W0, W0, W_s)
        if abs(lp) == np.inf:
            print('{} is inf'.format(labels[indx]))
            print(W)
            return lp, indx

    err_indx = 1000
    return lp, err_indx


##########  .  ##########



##########  Posterior function  ##########
def log_probability(theta, nu, power, target, modes, theta_init, bg_k, 
                    only_radial=False, fit_split=True, mixed_modes=True,
                    fit_bg=True, BG_bestFit=None):
    
    InputCatalog, _, _, _, _, _, _, _ = myF.get_target_info(target)
    
    lp, _ = log_prior(theta, nu, power, target, modes, theta_init, bg_k, 
                      only_radial=only_radial, fit_split=fit_split, 
                      mixed_modes=mixed_modes, fit_bg=fit_bg)
    
    
    return lp + log_likelihood(theta, nu, power, InputCatalog, modes, bg_k, 
                               only_radial=only_radial, fit_split=fit_split, mixed_modes=mixed_modes,
                               fit_bg=fit_bg, BG_bestFit=BG_bestFit)

##########  .  ##########



"""
###############################################################################

                        emcee sampler                      

###############################################################################
"""


def run_emcee(ndim, nwalkers, steps, p0, nu, pd, target, modes, theta_init, 
              discard_num, bg_k, PDSax, BGax, res_dir=None, multi_CPU=False, 
              only_radial=False, fit_split=True, mixed_modes=True,
              fit_bg=True, BG_bestFit=None):
    
    # check if priors are valid
    p_mean = np.mean(p0, axis=0)
    priors = []
    print('Testing intial guesses on priors')
    for p in p0:
        lp, err_indx = log_prior(p, nu, pd, target, modes, theta_init, bg_k,
                                 only_radial=only_radial, fit_split=fit_split, 
                                 mixed_modes=mixed_modes, fit_bg=fit_bg)
        
        loop_count = 0
        while err_indx < 1000:
            # print(p[err_indx])
            # print(p_mean[err_indx])
            p[err_indx] = p_mean[err_indx]
            print(p[err_indx])
            print('now changed')
            
            lp, err_indx = log_prior(p, nu, pd, target, modes, theta_init, bg_k,
                                     only_radial=only_radial, fit_split=fit_split, 
                                     mixed_modes=mixed_modes, fit_bg=fit_bg)
            
            loop_count += 1
            if loop_count > 99:
                err_indx = 1000
                print('Median value of walkers is not within the priors')
            
        if lp == -np.inf:
            priors.append(False)
            
        else:
            priors.append(True)
    
    
    
    if np.all(priors):
        print('Initial guesses are fine --> starting sampling')
        
        # set up backend
        sampler_file = 'PB_sampler.h5'
        backend = emcee.backends.HDFBackend(sampler_file)
        backend.reset(nwalkers, ndim)

        # run MCMC
        if multi_CPU:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                                pool=pool, backend=backend,
                                                args=(nu, pd, target, modes,
                                                      theta_init, bg_k,
                                                      only_radial, 
                                                      fit_split, mixed_modes,
                                                      fit_bg, BG_bestFit))
                
                sampler.run_mcmc(p0, steps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            backend=backend,
                                            args=(nu, pd, target, modes, 
                                                  theta_init, bg_k,
                                                  only_radial, 
                                                  fit_split, mixed_modes,
                                                  fit_bg, BG_bestFit))
            
            sampler.run_mcmc(p0, steps, progress=True)  
        
        
        
        PlotBestFit(PDSax, nu, pd, target, modes, bg_k, res_dir, BGax,
                    discard_num=discard_num, sampler=sampler,
                    only_radial=only_radial, 
                    fit_split=fit_split, mixed_modes=mixed_modes,
                    fit_bg=fit_bg, BG_bestFit=BG_bestFit)
        
        
    else:
        print('Initial guess is invalid within the priors. Try again or adjust the priors, initial guess and/or the distribution of walkers.')
        if err_indx == 1000:
            print('!! Median value of walkers is not within the priors !!')
        
        sampler = None
        
    return sampler
    

##########  .  ##########


"""
###############################################################################

                        Main function                     

###############################################################################
"""



def params_estimation(nu, pd, target, modes, steps, theta_init, unc, info_file, 
                      discard_num=0, bg_k=2, res_dir=None, multi_CPU=False, 
                      only_radial=False, fit_split=True, mixed_modes=True,
                      fit_bg=True, BG_bestFit=None):
    
    InputCatalog, ID, name, fmin, fmax, sigma, s_corr, s_numax = myF.get_target_info(target)
    
    plt.close('all') 
    # plot PDS
    myF.plot_PDS(nu, pd, smooth=True, sigma=sigma)#, plot_modes=True, modes=modes)
    
    # Define axis for later use
    PDSax = plt.gca()
    
    if fit_bg:
        myF.plot_PDS(nu, pd, smooth=True, sigma=500, loglog=True)
        BGax = plt.gca()
    else:
        BGax = None


    # perturbation such that each walkers have different starting position ... gaussian ball ...
    labels = get_labels(modes, fmin=nu.min(), fmax=nu.max(), only_radial=only_radial, 
                        fit_split=fit_split, mixed_modes=mixed_modes,
                        fit_bg=fit_bg, bg_k=bg_k)
    ndim = len(labels)
    
    
    nwalkers = 500
    #nwalkers = ndim*2
    info_file.write('\nWalkers: {}\n'.format(nwalkers))
    
    perturb = unc*np.random.randn(nwalkers, ndim)
    p0 = theta_init+perturb
    
    # plot walker distribution
    #InputCatalog, ID, name, _, _, _, _, _ = myF.get_target_info(target)
    
    for i, lab in enumerate(labels):
        fig, ax = plt.subplots(1, 1, figsize=(7,5))
        plt.subplots_adjust(top=0.92,
                            bottom=0.1,
                            left=0.1,
                            right=0.97,
                            hspace=0.5,
                            wspace=0.2)
        
        plt.hist(p0[:,i], bins=30)
        plt.axvline(theta_init[i], color='k', linestyle='dashed', linewidth=1,
                    label=r'$\theta_{init}$')
        plt.axvline(p0[:,i].mean(), color='r', linestyle='dashed', linewidth=1,
                    label='Mean value of walkers')
        plt.legend()
        
        plt.xlabel(lab+' (relevant units)')
        plt.ylabel('Counts')
        
        #fig.suptitle('{} {} ({})'.format(InputCatalog, ID, name))
        ax.text(1, 1, name, transform=ax.transAxes,
            va='bottom', ha='right', fontsize=10)
            
        bad_chars = ['$', '\\', '{', '}']
        for i in bad_chars:
            lab = lab.replace(i, '')
        
        fig.savefig(res_dir+'/walker_distribution_'+lab+'.png', dpi=300)
        plt.close(fig)
    
    
    # run the sampler
    sampler = run_emcee(ndim, nwalkers, steps, p0, nu, pd, 
                        target, modes, theta_init, 
                        discard_num, bg_k, PDSax, BGax, 
                        res_dir=res_dir, multi_CPU=multi_CPU,
                        only_radial=only_radial, 
                        fit_split=fit_split, mixed_modes=mixed_modes,
                        fit_bg=fit_bg, BG_bestFit=BG_bestFit)
    
    
    return sampler

##########  .  ##########

