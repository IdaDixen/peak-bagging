#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:21:26 2023

@author: Ida Dixen Skaarup Wölm


###############################################################################

                        Background modelling with MCMC                         

###############################################################################


"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scipy.integrate as integrate

from multiprocessing import Pool

import emcee
import corner

import functions as myF
from priors import uniform_prior, Jeffreys_prior, mod_Jeffreys_prior, truncated_Gaussian_prior



"""
###############################################################################

                               Parameter control                               

###############################################################################
"""

def get_labels(k=2, gaussian_env=True, alt2=False):
    
    if gaussian_env:
        labels = [r'$\nu_{max}$', r'$A_{max}$', r'$\sigma_{env}$'] 
    else:
        labels = []

    for i in range(k):
        if (k == 3) | alt2:
            i -= 1
        labels.append(r'$\sigma_{}$'.format(i+1))
    
    for i in range(k):
        if (k == 3) | alt2:
            i -= 1
        labels.append(r'$\tau_{}$'.format(i+1))
    
    labels.append(r'W')    

    return labels


def get_index(k=2):
    
    k_max = k   # number of bg components
    
    W_index = -1
    tau_max = W_index
    tau_min = tau_max-k_max
    sigma_max = tau_min
    sigma_min = tau_min-k_max
    
    index = sigma_min, sigma_max, tau_min, tau_max, W_index
    
    return index


def unpack_theta(theta, k=2, gaussian_env=True): 
    # nu_max, A_max, sigma_env, ..., sigma_k, ..., tau_k, ..., W

    #### BACKGROUND ####
    sigma_min, sigma_max, tau_min, tau_max, W_index = get_index(k)
    
    sigma_k = theta[sigma_min:sigma_max]
    tau_k   = theta[tau_min:tau_max]
    W       = theta[W_index]
    
    
    if gaussian_env:
        ####  Power envelope  ####
        nu_max      = theta[0]
        A_max       = theta[1]
        sigma_env   = theta[2]
        
        return nu_max, A_max, sigma_env, sigma_k, tau_k, W
    
    else:
        return sigma_k, tau_k, W





"""
###############################################################################

                         Bayesian inference modelling

###############################################################################
"""

##########  Models  ##########

def bg_model(nu, sigma_k, tau_k, components=False):
    
    xi_k = 4*np.sqrt(2)
    
    N_nu = 0
    N_components = []
    for sigma, tau in zip(sigma_k, tau_k):
        N_k = ( xi_k * sigma**2 * tau) / (1 + (2*np.pi * nu * tau)**4)
        N_nu += N_k
        N_components.append(N_k)
    
    if components:
        return N_nu, N_components
    else:
        return N_nu


def model(theta, nu, k, InputCatalog):
    
    nu_max, A_max, sigma_env, sigma_k, tau_k, W = unpack_theta(theta, k)
    
    ####  Apodization  ####
    if InputCatalog == 'KIC':
        Delta_t = 58.89
    elif InputCatalog == 'TIC':
        Delta_t = 120
    
    x = np.pi * nu/1e6 * Delta_t
    eta = np.sin(x)/x
    
    
    ####  BACKGROUND  ####
    B = bg_model(nu, sigma_k, tau_k)
    
    
    ####  Power envelope  ####
    P = A_max * np.exp( -(nu-nu_max)**2 / (2*sigma_env**2) )
       
    return eta**2 * (P + B) + W


##########  .  ##########



##########  Likelihood function  ##########

def log_likelihood(theta, nu, power, k, InputCatalog):
    
    model_fit = model(theta, nu, k, InputCatalog)
    
    return -np.nansum(np.log(model_fit) + power/model_fit)

##########  .  ##########



##########  Priors  ##########

def log_prior(theta, nu, pd, target, theta_init, k):
    
    nu_max, A_max, sigma_env, sigma_k, tau_k, W = unpack_theta(theta, k)
    numax_init, Amax_init, sigma_env_init, sigma_k_init, tau_k_init, W_init = unpack_theta(theta_init, k)
    
    Dnu = myF.get_Dnu(target)
    labels = get_labels(k)
 
    
    # add since they are log!
    
    indx = 0
    lp = uniform_prior(nu_max, numax_init-4*Dnu, numax_init+4*Dnu)
    if abs(lp) == np.inf:
        print('{} is inf'.format(labels[indx]))
        print(nu_max)
        return lp, indx
    
    indx += 1
    lp += mod_Jeffreys_prior(A_max, 0, max(pd)/2, Amax_init)
    if abs(lp) == np.inf:
        print('{} is inf'.format(labels[indx]))
        print(A_max)
        return lp, indx
    
    indx += 1
    lp += Jeffreys_prior(sigma_env, 0.05*numax_init, 0.3*numax_init)
    if abs(lp) == np.inf:
        print('{} is inf'.format(labels[indx]))
        print(sigma_env)
        return lp, indx

    
    for sigma, sigma_init in zip(sigma_k, sigma_k_init):
        indx += 1
        lp += mod_Jeffreys_prior(sigma, 0, np.sqrt(integrate.simps(pd,nu)), sigma_init)
        if abs(lp) == np.inf:
            print('{} is inf'.format(labels[indx]))
            print(sigma)
            return lp, indx

    
    tau_lower = [1/(2*np.pi*numax_init*0.3), 1/(2*np.pi*numax_init*0.7), 1/(2*np.pi*numax_init*2)]
    tau_upper = [1/(2*np.pi*numax_init*0.01), 1/(2*np.pi*numax_init*0.1), 1/(2*np.pi*numax_init*0.7)]
    if k == 1:
        tau = tau_k
        tau_init = tau_k_init
        indx += 1
        lp += mod_Jeffreys_prior(tau, tau_lower[-1], tau_upper[0], tau_init)
        if abs(lp) == np.inf:
            print('{} is inf'.format(labels[indx]))
            print(tau)
            return lp, indx
    else:
        for ii, (tau, tau_init) in enumerate(zip(tau_k, tau_k_init)):
            indx += 1
            if k == 2:
                ii += 1
            lp += mod_Jeffreys_prior(tau, tau_lower[ii], tau_upper[ii], tau_init)
            if abs(lp) == np.inf:
                print('{} is inf'.format(labels[indx]))
                print(tau)
                return lp, indx
    
    indx += 1
    lp += uniform_prior(W, 0, W_init*100)
    if abs(lp) == np.inf:
        print('{} is inf'.format(labels[indx]))
        print(W)
        return lp, indx
    
    
    err_indx = 1000
    return lp, err_indx

##########  .  ##########



##########  Posterior function  ##########
def log_probability(theta, nu, power, target, theta_init, k):
    
    InputCatalog, _, _, _, _, _, _, _ = myF.get_target_info(target)
    
    lp, _ = log_prior(theta, nu, power, target, theta_init, k)
    
    return lp + log_likelihood(theta, nu, power, k, InputCatalog)
##########  .  ##########


"""
###############################################################################

                        Plotting                      

###############################################################################
"""
def get_chain(sampler, labels, discard_num, res_dir=None, plot=True):
    
    if plot:
        samples = sampler.get_chain()
        plot_dim = len(labels)
        
        fig, axes = plt.subplots(plot_dim, figsize=(12, 12), sharex=True)
        for j in range(plot_dim):
            ax = axes[j]
            ax.plot(samples[:, :, j], 'k', alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[j])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        
        if res_dir is not None:
            fig.savefig(res_dir+'/BG_ChainPlot.png', dpi=300)
            
        plt.close()
        
    # calculate flat samples
    flat_samples = sampler.get_chain(discard=discard_num, thin=1, flat=True)
    
    return flat_samples


def cornerPlot(sampler, nu, discard_num=0, k=2, res_dir=None, filename=None, 
               chain=True, plot=True):
    
    labels = get_labels(k)
    flat_samples = get_chain(sampler, labels, discard_num, res_dir, plot=chain)
    
    if plot:
        
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        f = corner.corner(flat_samples, 
                          labels=labels, 
                          bins=30,
                          color='teal',
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True,
                          label_kwargs=dict(fontsize=20),
                          title_kwargs=dict(fontsize=10),
                          hist_kwargs={'histtype': 'stepfilled', 'alpha': 0.4},
                          levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                          plot_datapoints=True,
                          fill_contours=True,
                          max_n_ticks=4,
                          title_fmt='.2e',
                          )
        
        for ax in f.get_axes():
            formatter = ScalarFormatter()
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2,3))
            if ax.get_xlabel() != '':
                ax.xaxis.set_major_formatter(formatter)
            if ax.get_ylabel() != '':
                ax.yaxis.set_major_formatter(formatter)
                ax.yaxis.get_offset_text().set_position((-.2, -2))
        
        if res_dir is not None:
            ax = plt.gca()
            ax.figure.savefig(res_dir+'/BG_CornerPlot.png', dpi=300)
        
        plt.close()
        
    
    best_theta = []
    lower = []
    upper = []
     
    ndim = len(labels)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        best_theta = np.append(best_theta, mcmc[1])
        lower = np.append(lower, q[0])
        upper = np.append(upper, q[1])
    
    if res_dir is not None:
        results = np.column_stack((labels, best_theta, lower, upper))
        np.savetxt(res_dir+'/'+filename, results, delimiter='\t', fmt='%s')
    
    return best_theta


def PlotBestFit(ax, nu, target, res_dir=None, discard_num=0, k=2, 
                sampler=None, best_theta=None, 
                chain_plot=True, corner_plot=True, alt2=False):
    
    if sampler is None and best_theta is None:
        print('Error. Either sampler or best_theta has to be specified')
        return None

    
    InputCatalog, ID, name, _, _, _, _, _ = myF.get_target_info(target) 

    if sampler is not None:
        filename = 'BG_BestFitResults_{}{}.txt'.format(InputCatalog, ID)
        best_theta = cornerPlot(sampler, nu, discard_num, k, res_dir, filename, 
                                chain=chain_plot, plot=corner_plot)
        
    #### BACKGROUND ####
    sigma_min, sigma_max, tau_min, tau_max, W_index = get_index(k)
    
    sigma_k = best_theta[sigma_min:sigma_max]
    tau_k   = best_theta[tau_min:tau_max]
    W       = best_theta[W_index]

    N_nu, N_comps = bg_model(nu, sigma_k, tau_k, components=True)
    bg_fit = N_nu+W
    P_fit = model(best_theta, nu, k, InputCatalog)
    
    styles = ['dotted', 'dashdot', 'solid']
    for i, N_comp in enumerate(N_comps):
        ii = i
        if (k == 3) | alt2:
            ii -= 1
        ax.loglog(nu, N_comp, linestyle=styles[ii], color='lime', 
                  label=str(ii+1)+r'. comp. of $\mathcal{B}$')
    ax.hlines(W, nu.min(), nu.max(), linestyle='dashed', color='lime', 
              label=r'$W$')
    ax.loglog(nu, bg_fit, 'r', label=r'Total $\mathcal{B}$')
    ax.loglog(nu, P_fit, 'r', linestyle='dashed', 
              label=r'$\mathcal{P}_{Gauss}$')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

    if res_dir is not None:
        ax.figure.savefig(res_dir+'/BG_bestFit.png', dpi=300)
            
    
    return None

##########  .  ##########



"""
###############################################################################

                        emcee sampler                      

###############################################################################
"""

def run_emcee(ndim, nwalkers, steps, p0, nu, pd, target, theta_init, 
              discard_num, k, ax, res_dir=None, multi_CPU=False):

    # check if priors are valid
    p_mean = np.mean(p0, axis=0)
    priors = []
    print('Testing intial guesses on priors')
    for p in p0:
        lp, err_indx = log_prior(p, nu, pd, target, theta_init, k)
        
        loop_count = 0
        while err_indx < 1000:
            print(p[err_indx])
            print(p_mean[err_indx])
            p[err_indx] = p_mean[err_indx]
            print(p[err_indx])
            print('now changed')
            
            lp, err_indx = log_prior(p, nu, pd, target, theta_init, k)
            
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
        sampler_file = 'BG_sampler.h5'
        backend = emcee.backends.HDFBackend(sampler_file)
        backend.reset(nwalkers, ndim)
    
        # run MCMC
        if multi_CPU:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                                pool=pool, backend=backend,
                                                args=(nu, pd, target, theta_init, k))
                
                sampler.run_mcmc(p0, steps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            backend=backend,
                                            args=(nu, pd, target, theta_init, k))
            
            sampler.run_mcmc(p0, steps, progress=True)  
        
        
        PlotBestFit(ax, nu, target, res_dir, 
                    discard_num=discard_num, k=k, sampler=sampler)
        
    else:
        print('Initial guess is invalid within the priors. Try again or adjust the priors, initial guess or/and the distribution of walkers.')
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


def params_estimation(nu, pd, target, steps, theta_init, unc, discard_num=0, 
                      k=2, res_dir=None, multi_CPU=False):
    
    InputCatalog, ID, name, _, _, _, _, s_numax = myF.get_target_info(target)
    
    plt.close('all')
    
    myF.plot_PDS(nu, pd, smooth=True, sigma=s_numax, loglog=True)
    ax = plt.gca()
    ax.figure.suptitle('Background fit for {} {} ({})'.format(InputCatalog, ID, name))


    # perturbation such that each walkers have different starting position ... gaussian ball ...
    labels = get_labels(k)
    ndim = len(labels)

    nwalkers = 500
    #nwalkers = 4*ndim
    
    perturb = unc*np.random.randn(nwalkers, ndim)
    p0 = theta_init+perturb
    
    # plot walker distribution
    InputCatalog, ID, name, _, _, _, _, _ = myF.get_target_info(target)
    for i, lab in enumerate(labels):
        fig1, ax1 = plt.subplots(1, 1, figsize=(7,5))
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
        
        plt.xlabel(lab+' relevant units')
        plt.ylabel('Counts')
        
        fig1.suptitle('{} {} ({})'.format(InputCatalog, ID, name))
        
            
        bad_chars = ['$', '\\', '{', '}']
        for i in bad_chars:
            lab = lab.replace(i, '')
        
        fig1.savefig(res_dir+'/walker_distribution_'+lab+'.png', dpi=300)
        plt.close(fig1)
    
    
    sampler = run_emcee(ndim, nwalkers, steps, p0, nu, pd, target, 
                        theta_init, discard_num, k, ax, 
                        res_dir=res_dir, multi_CPU=multi_CPU)

    return sampler

##########  .  ##########

