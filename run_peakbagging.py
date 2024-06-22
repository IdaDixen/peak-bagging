#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 4 09:55:40 2023

@author: Ida Dixen Skaarup Wölm




###############################################################################

                     Peakbagging of solar-like oscillators                     

###############################################################################


PREREQUISITS

Before run, make sure the following is changed to fit the run:
    -> target (if not on Grendel)
    -> test_info
If another filetype is needed, change pdsfile

Other settingd to consider:
    -> step_num
    -> discard_num
    -> mk_dir
    -> mCPU
    -> only_radial
    -> fit_bg
    -> fit_split
    -> mixed_modes
    -> F-like
    -> simple_guess

Target info must be contained in: myF.get_target_info(target)
Morover the following files are needed: (have to be stored in MyFunctions folder)
    -> Modes (Folder with modes_{}{}.csv'.format(InputCatalog, ID) is accepted)
    -> 'theta_init/theta_init.csv' including uncertainties (in separate uncertainties/ folder) 
        to make Gaussian ball to radnomize walker positions
    -> theta_init/BG_BestFitResults_{}{}.txt'.format(InputCatalog, ID)
    -> Find_peaks/'Dnus.csv'
    
If the model is changed, remember to change priors, theta_init and unc as well



"""


import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('MyFunctions')
import functions as myF
import MCMC_modes as mcmc

import os
from datetime import datetime
import time
start = time.time()


def P_fit(a, b, numax):
    P = a*(numax/3090)+b
    return P



#####  Initial settings  #####

mk_dir          = True
mCPU            = True

step_num        = 2000
discard_num     = 500

only_radial     = False
fit_split_arr   = [True, True, True, False, False, False]
mixed_modes_arr = [True, True, True, True, False, False]
F_like          = False
fit_bg          = True
simple_geuss    = False



target_list = myF.get_target_list()
target_indx = int(sys.argv[1])
target = target_list[target_indx]
InputCatalog, ID, name, fmin, fmax, sigma, s_corr, s_numax = myF.get_target_info(target)
fit_split = fit_split_arr[target_indx]
mixed_modes = mixed_modes_arr[target_indx]

test_info = 'run tess targets again with new split handling in model'

print(test_info)
print('Target: {}{} ({})'.format(InputCatalog, ID, name))

# make res dir if defined:
if mk_dir:
    
    res_dir = 'PB_results_'+InputCatalog+ID+'_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mydir = os.path.join(
            os.getcwd(), 
            res_dir)
    
    os.makedirs(mydir)
else:
    res_dir=None


#####  Read PDS file  #####

print('Loading data')

pdsfile = 'MyFunctions/Data/wpds_{}{}.csv'.format(InputCatalog, ID)
data = np.loadtxt(pdsfile, delimiter=',')
nu = data[:,0]
pd = data[:,1]

mask = np.where((nu>fmin) & (nu<fmax))[0]
nu = nu[mask]
pd = pd[mask]

if F_like:
    mixed_modes = False
    mode_dir = 'MyFunctions/Modes/modes_{}{}_centroids.csv'.format(InputCatalog, ID)
else:    
    mode_dir = 'MyFunctions/Modes/modes_{}{}.csv'.format(InputCatalog, ID)
modes = myF.get_modes(mode_dir)
Dnu = myF.get_Dnu(target)




#####  Initial guesses  #####
print('Defining intial parameter guess')

BG_file = 'MyFunctions/theta_init/BG_BestFitResults_{}{}.txt'.format(InputCatalog, ID)
BG_results = Table.read(BG_file, format='ascii')
bg_k = int((len(BG_results)-4)/2)
BG_bestFit=None


l0_modes = modes[(modes['l']==0)]
l1_modes = modes[(modes['l']==1)]
l01_modes = modes[(modes['l']==0) | (modes['l']==1)]

if only_radial | (not mixed_modes and not fit_split):
    fit_modes = l0_modes
else:
    fit_modes = l01_modes

if only_radial:
    modes['nu'][l01_modes]
    


if not simple_geuss:
    nu_max, A_env, sigma_env = BG_results['col2'][0:3]
    
    Amax = -(3/250)*nu_max+18
    Amin = 0.1*Amax
    
    # mode closest to nu_max:
    nearest_numax = np.argmin(abs(fit_modes['nu']-nu_max))
    nu_left = fit_modes['nu'][:nearest_numax+1]
    nu_right = fit_modes['nu'][nearest_numax:]
    
    ## Amplitudes ###
    A1 = np.interp(nu_left, [min(nu_left), max(nu_left)], [Amin, Amax])
    A2 = np.interp(nu_right, [min(nu_right), max(nu_right)], [Amax, Amin])
    A_guess = np.append(A1[:-1], A2)
    
    # Widths
    alpha = P_fit(2.95, 0.39, nu_max)
    Gamma_alpha = P_fit(3.08, 3.32, nu_max)
    DGamma_dip = P_fit(-0.47, 0.62, nu_max)
    nu_dip = P_fit(2984, 60, nu_max)
    W_dip = P_fit(4637, -141, nu_max)
    lnGamma = (alpha*np.log(nu/nu_max))+np.log(Gamma_alpha) + ((np.log(DGamma_dip))/(1+((2*np.log(nu/nu_dip))/(np.log(W_dip/nu_max)))**2))
    
    Gamma_indx = np.searchsorted(nu,fit_modes['nu'])
    Gamma_guess = np.exp(lnGamma[Gamma_indx])
    
    
    # plot A and Gamma init
    fig, ax1 = plt.subplots(1, 1, figsize=(7,4))
    ax2 = ax1.twinx()
    plt.subplots_adjust(top=0.92,
                        bottom=0.125,
                        left=0.1,
                        right=0.8,
                        hspace=0.5,
                        wspace=0.2)
    
    lns1 = ax1.plot(fit_modes['nu'], A_guess, '.-', linewidth=.3, 
                    label=r'$A_{n\ell,init}$')
    ax1.set_xlabel(r'$\nu$ [$\mu$Hz]')
    ax1.set_ylabel(r'$A$ [ppm]')
    
    lns2 = ax2.semilogy(fit_modes['nu'], Gamma_guess, 'r.-', linewidth=.3,
                        label=r'$\Gamma_{n\ell,init}$')
    ax2.set_ylabel(r'$\Gamma$ [$\mu$Hz]')
    
    lns3 = ax1.vlines(nu_max, Amin-.2, Amax+.2, 'k', linestyle='dotted',
                      linewidth=.8, label=r'$\nu_{max}$')
    
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    if mk_dir:
        fig.savefig(res_dir+'/A_Gamma_init.png', dpi=300)
   


####################################################################

# add nu, A and gamma guess' and unc

if simple_geuss:
    simple_Aguess = np.sqrt(max(pd)/2)
    
    theta_init = np.concatenate((modes['nu'],
                                simple_Aguess*np.ones(len(fit_modes)),  #A_guess
                                1*np.ones(len(fit_modes))   #Gamma_guess
                                ))
    
    unc = np.concatenate((#.1*Dnu
                          0.5*np.ones(len(modes)), 
                          0.1*simple_Aguess*np.ones(len(fit_modes)),
                          .1*np.ones(len(fit_modes)))
                        )

else:
    theta_init = np.concatenate((modes['nu'],
                                A_guess,
                                Gamma_guess,
                                ))

    unc = np.concatenate((#.1*Dnu*
                          .5*np.ones(len(modes)), 
                          .5*np.ones(len(fit_modes)),
                          .1*np.ones(len(fit_modes)))
                        )




if not only_radial:
    # add V_l, nu_s and i
    
    # degrees present
    ls = np.unique(modes['l'])
    
    # Visibility guess
    # V is refering to V^tilde_l = V_l/V_0
    Vl_guess = [1.22, 0.71, 0.14]
    Vl_unc = [0.01, 0.01, 0.01]
    
    # Initial guesses if splitting
    if fit_split:
        # rotation guess
        nu_s_guess = [0.3, 0.3, 0.5, 0, 0, 0][target_indx]
        nu_s_unc = 0.05
        deg = [40, 60, 20, 0, 0, 0][target_indx]
        i_guess = np.cos(np.deg2rad(deg))
        i_unc = 0.1
        l_maxfit = 1
    else:
        l_maxfit = 0
      
        
    # deal with mixed modes
    if mixed_modes:
        l_maxfit = 1
        
        # add nuiscent visibility per order for l=1 mixed modes
        theta_init = np.append(theta_init, Vl_guess[0]*np.ones(len(l1_modes)))
        unc = np.append(unc, Vl_unc[0]*np.ones(len(l1_modes)))
        
        # add visibility for interpolated modes
        for l in ls[ls>l_maxfit]:
            indx = l-1
            theta_init = np.append(theta_init, Vl_guess[indx])
            unc = np.append(unc, Vl_unc[indx])
        
        
        if fit_split:
            theta_init = np.concatenate((theta_init, 
                                         nu_s_guess*np.ones(len(l1_modes)), 
                                         [i_guess]))
            unc = np.concatenate((unc, 
                                  nu_s_unc*np.ones(len(l1_modes)), 
                                  [i_unc]))
            
    else: # if no mixed modes
        
        # add visibility for interpolated modes
        for l in ls[ls>l_maxfit]:   # if no splitting, then l_maxfit = 0 else 1
            indx = l-1
            theta_init = np.append(theta_init, Vl_guess[indx])
            unc = np.append(unc, Vl_unc[indx])
        
        
        if fit_split:
            # add nu_s and i
            theta_init = np.append(theta_init, [nu_s_guess, i_guess])
            unc = np.append(unc, [nu_s_unc, i_unc])

    
if fit_bg:
    # bg_k = int((len(BG_results)-4)/2)
    sigmas = BG_results['col2'][3:3+bg_k]
    sigma_uncs = 0.05*sigmas
    theta_init = np.append(theta_init, sigmas)
    unc = np.append(unc, sigma_uncs)

    taus = BG_results['col2'][3+bg_k:3+2*bg_k]
    tau_uncs = 0.05*taus
    theta_init = np.append(theta_init, taus)
    unc = np.append(unc, tau_uncs)
    
    W = BG_results['col2'][-1]
    Wunc = 0.05*W
    theta_init = np.append(theta_init, W)
    unc = np.append(unc, Wunc)
    
    BG_bestFit = None
    
else:
    BG_bestFit = list(BG_results['col2'][3:])





#####  MCMC sampling #####



print('Starting sampling ({} parameters)\n\n'.format(len(theta_init)))


info_file = open(res_dir+'/run_info.txt', 'w')
info_file.write('Target: {}{}\n'.format(InputCatalog, ID))
info_file.write('Test Info: {}\n\n'.format(test_info))

info_file.write('\nfmin = {}\n'.format(fmin))
info_file.write('fmax = {}\n'.format(fmax))

info_file.write('\nStellar settings:\n')
info_file.write('    Only radial fitted: {}\n'.format(only_radial))
info_file.write('    Fitting rotaitonal split: {}\n'.format(fit_split))
info_file.write('    Fitting mixed modes: {}\n'.format(mixed_modes))
info_file.write('    An F-like star fit: {}\n'.format(F_like))
info_file.write('    Background fit: {}\n'.format(fit_bg))
info_file.write('    #background components: {}\n'.format(bg_k))

info_file.write('\nResults directory: {}\n'.format(res_dir))
info_file.write('Parallelization: {}\n'.format(mCPU))
info_file.write('Dimensions (# of parameters): {}\n'.format(len(theta_init)))



sampler = mcmc.params_estimation(nu, pd, target, modes, step_num, 
                                 theta_init, unc, info_file, 
                                 discard_num=discard_num, bg_k=bg_k,
                                 res_dir=res_dir, multi_CPU=mCPU,
                                 only_radial=only_radial, 
                                 fit_split=fit_split, mixed_modes=mixed_modes,
                                 fit_bg=fit_bg, BG_bestFit=BG_bestFit)



#####  Save run info  #####

end = time.time()
run_time = end - start
unit = 'sec'

if run_time > 60:
    run_time = run_time/60
    unit = 'min'

    if run_time > 60:
        run_time = run_time/60
        unit = 'hours'

        if run_time > 24:
            run_time = run_time/24
            unit = 'days'

info_file.write('# of steps/iterations: {}\n'.format(step_num))
info_file.write('# of steps discarded from the beginning of the chain: {}\n'.format(discard_num))
info_file.write('Run time: {} {}.'.format(run_time, unit))
info_file.close()

if sampler is None:
    print('Peakbagging was not initiated')
else:
    print('\n\nPeakbagging has completed. Run time: {} {}.\n'.format(run_time, unit))

