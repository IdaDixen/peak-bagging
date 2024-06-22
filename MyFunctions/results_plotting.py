#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:02:11 2024

@author: idadixenskaarupwolm
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import corner

import MCMC_modes as ps
import MCMC_bg as bg
import functions as myF


"""
###############################################################################

                        Plotting                      

###############################################################################
"""

def get_chain(sampler, labels, discard_num, modes, bg_k, res_dir=None, 
              plot=True, only_radial=False, fit_split=True, mixed_modes=True, 
              fit_bg=True):
    
    
    # heights and widths
    A_min, A_max, Gamma_min, Gamma_max = ps.get_index('fit', modes, 
                                                      fit_split=fit_split, 
                                                      mixed_modes=mixed_modes)
        
    if not only_radial:
        
        if mixed_modes:
            V_nl_min, V_nl_max, V_l_min, V_l_max = ps.get_index('interpolate', 
                                                                 modes,  
                                                                 fit_split=fit_split, 
                                                                 mixed_modes=mixed_modes)
            
            if fit_split:
                nu_s_min, nu_s_max, i_index = ps.get_index('split', modes,  
                                                           fit_split=fit_split, 
                                                           mixed_modes=mixed_modes)
        
        else: # if no mixed modes
            V_l_min, V_l_max = ps.get_index('interpolate', modes,  
                                            fit_split=fit_split, 
                                            mixed_modes=mixed_modes)
            
            if fit_split:
                nu_s_index, i_index = ps.get_index('split', modes,  
                                                   fit_split=fit_split, 
                                                   mixed_modes=mixed_modes)
    
    
    if plot:
        samples = sampler.get_chain()
        
        samples[:,:,A_min:A_max] = abs(samples[:,:,A_min:A_max])
        #if fit_split:
        #    u_mask = samples[:,:,i_index]>1
        #    samples[:,:,i_index][u_mask] = 1-(samples[:,:,i_index][u_mask]-1)
        #    if mixed_modes:
        #        samples[:,:,nu_s_min:nu_s_max] = abs(samples[:,:,nu_s_min:nu_s_max])
        #    else:
        #        samples[:,:,nu_s_index] = abs(samples[:,:,nu_s_index])
        
        
        if fit_split | mixed_modes:
            mask_fit = (modes['l']==0) | (modes['l']==1)
        else:
            mask_fit = (modes['l']==0)
        
        Nmodes = len(modes)
        
        nu_nl     = samples[:,:,:Nmodes][:,:,mask_fit]
        A_nl      = samples[:,:,A_min:A_max]
        Gamma_nl  = samples[:,:,Gamma_min:Gamma_max]
        
        nu_labels       = labels[:Nmodes][mask_fit]
        A_labels        = labels[A_min:A_max]
        Gamma_labels    = labels[Gamma_min:Gamma_max]
        
        if not only_radial:
            if mixed_modes:
                
                V_nl     = samples[:,:,V_nl_min:V_nl_max]
                V_nl_labels    = labels[V_nl_min:V_nl_max]
                
                if (V_l_min-V_l_max) == 0:
                    V_l     = samples[:,:,V_l_min]
                    V_l_labels    = labels[V_l_min]
                else:
                    V_l     = samples[:,:,V_l_min:V_l_max]
                    V_l_labels    = labels[V_l_min:V_l_max]
                
                if fit_split:
                    nu_s    = samples[:,:,nu_s_min:nu_s_max]
                    i       = samples[:,:,i_index]
                    #i = np.rad2deg(np.arccos(i))
                    
                    nu_s_labels = labels[nu_s_min:nu_s_max]
                    i_label     = labels[i_index]
            else:
                if (V_l_min-V_l_max) == 0:
                    V_l     = samples[:,:,V_l_min]
                    V_l_labels    = labels[V_l_min]
                else:
                    V_l     = samples[:,:,V_l_min:V_l_max]
                    V_l_labels    = labels[V_l_min:V_l_max]
                
                if fit_split:
                    nu_s    = samples[:,:,nu_s_index]
                    i       = samples[:,:,i_index]
                    #i = np.rad2deg(np.arccos(i))
                    
                    nu_s_labels = labels[nu_s_index]
                    i_label     = labels[i_index]
            
        if fit_bg:
            sigma_min, sigma_max, tau_min, tau_max, W_index = bg.get_index(bg_k)
            
            if (sigma_min-sigma_max) == 0:
                sigma_k = samples[:,:,sigma_min]
                tau_k   = samples[:,:,tau_min]
            else:
                sigma_k = samples[:,:,sigma_min:sigma_max]
                tau_k   = samples[:,:,tau_min:tau_max]
            W       = samples[:,:,W_index]
            
            if (sigma_min-sigma_max) == 0:
                sigma_k_lab = labels[sigma_min]
                tau_k_lab   = labels[tau_min]
            else:
                sigma_k_lab = labels[sigma_min:sigma_max]
                tau_k_lab   = labels[tau_min:tau_max]
            W_lab       = labels[W_index]
            
            plot_dim = abs(sigma_min)
            plot_params = np.dstack((sigma_k, tau_k, W)).T
            labels_short = np.concatenate((sigma_k_lab, tau_k_lab, [W_lab]))
            
            fig, axes = plt.subplots(plot_dim, figsize=(10, 7), sharex=True)
            for j in range(plot_dim):
                ax = axes[j]
                ax.plot(plot_params[j].T, 'k', alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels_short[j])
                ax.yaxis.set_label_coords(-0.1, 0.5)
    
            axes[-1].set_xlabel("step number")
            
            if res_dir is not None:
                fig.savefig(res_dir+'/BG_ChainPlot.png', dpi=300)
            
            plt.close()
        
        
        l1_indx = -1
        loop_values = zip(nu_nl.T, A_nl.T, Gamma_nl.T, nu_labels, A_labels, Gamma_labels)
        for nl, (nu, A, Gamma, nu_lab, A_lab, Gamma_lab) in enumerate(loop_values):
            
            plot_dim = 3
            plot_params = [nu.T, A.T, Gamma.T]
            labels_short = [nu_lab, A_lab, Gamma_lab]
            
            if not only_radial:
                current_l = modes[mask_fit][nl]['l'] 
                
                if current_l == 1:
                    l1_indx += 1
                
                if mixed_modes:
                    if current_l == 1:
                        plot_dim += 1
                        plot_params.append(V_nl[:,:,l1_indx])
                        labels_short.append(V_nl_labels[l1_indx])
                    
                if fit_split & (current_l == 1):
                    plot_dim += 2
                    
                    if mixed_modes:
                        plot_params.append(nu_s[:,:,l1_indx])
                        labels_short.append(nu_s_labels[l1_indx])
                    else:
                        plot_params.append(nu_s)
                        labels_short.append(nu_s_labels)
                    
                    plot_params.append(i)
                    labels_short.append(i_label)
                
                # V_l implementation
                if current_l == 0:
                    if fit_split | mixed_modes:
                        l_maxfit = 1
                    else:
                        l_maxfit = 0
                    
                    ls = np.unique(modes['l'])
                    plot_dim += len(ls[ls>l_maxfit])
                    for V, V_lab in zip(V_l.T, V_l_labels):
                        plot_params.append(V.T)
                        labels_short.append(V_lab)
            
            fig, axes = plt.subplots(plot_dim, figsize=(10, 7), sharex=True)
            for j in range(plot_dim):
                ax = axes[j]
                ax.plot(plot_params[j], 'k', alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels_short[j])
                ax.yaxis.set_label_coords(-0.1, 0.5)
    
            axes[-1].set_xlabel('step number')
            
            if res_dir is not None:
                bad_chars = ['$', '\\', '{', '}']
                for j in bad_chars:
                    nu_lab = nu_lab.replace(j, '')
                    
                fig.savefig(res_dir+'/ChainPlot_{}.png'.format(nu_lab), dpi=300)
            
            plt.close()
            
    
    # calculate flat samples
    flat_samples = sampler.get_chain(discard=discard_num, thin=1, flat=True)
    
    flat_samples[:,A_min:A_max] = abs(flat_samples[:,A_min:A_max])
    #if fit_split:
    #    u_mask = flat_samples[:,i_index]>1
    #    flat_samples[:,i_index][u_mask] = 1-(flat_samples[:,i_index][u_mask]-1)
    #    if mixed_modes:
    #        flat_samples[:,nu_s_min:nu_s_max] = abs(flat_samples[:,nu_s_min:nu_s_max])
    #    else:
    #        flat_samples[:,nu_s_index] = abs(flat_samples[:,nu_s_index])
    
    return flat_samples


def cornerPlot(sampler, nu, modes, bg_k, discard_num=0,
               res_dir=None, chain=True, plot=True, only_radial=False, 
               fit_split=True, mixed_modes=True, fit_bg=True):
    
    labels = ps.get_labels(modes, fmin=nu.min(), fmax=nu.max(), only_radial=only_radial, 
                           fit_split=fit_split, mixed_modes=mixed_modes,
                           fit_bg=fit_bg, bg_k=bg_k)

    flat_samples = get_chain(sampler, labels, discard_num, modes, bg_k, res_dir, 
                             plot=chain, only_radial=only_radial, 
                             fit_split=fit_split, mixed_modes=mixed_modes, fit_bg=fit_bg)
    
    
    if plot:
        
        # heights and widths
        A_min, A_max, Gamma_min, Gamma_max = ps.get_index('fit', modes, 
                                                          fit_split=fit_split, 
                                                          mixed_modes=mixed_modes)
            
        if not only_radial:
            
            if mixed_modes:
                V_nl_min, V_nl_max, V_l_min, V_l_max = ps.get_index('interpolate', 
                                                                     modes,  
                                                                     fit_split=fit_split, 
                                                                     mixed_modes=mixed_modes)
                
                if fit_split:
                    nu_s_min, nu_s_max, i_index = ps.get_index('split', modes,  
                                                               fit_split=fit_split, 
                                                               mixed_modes=mixed_modes)
            
            else: # if no mixed modes
                V_l_min, V_l_max = ps.get_index('interpolate', modes,  
                                                fit_split=fit_split, 
                                                mixed_modes=mixed_modes)
                
                if fit_split:
                    nu_s_index, i_index = ps.get_index('split', modes,  
                                                       fit_split=fit_split, 
                                                       mixed_modes=mixed_modes)
        
        if fit_split | mixed_modes:
            mask_fit = (modes['l']==0) | (modes['l']==1)
        else:
            mask_fit = (modes['l']==0)
        
        Nmodes = len(modes)
        
        nu_nl     = flat_samples[:,:Nmodes][:,mask_fit]
        A_nl      = flat_samples[:,A_min:A_max]
        Gamma_nl  = flat_samples[:,Gamma_min:Gamma_max]

        nu_labels       = labels[:Nmodes][mask_fit]
        A_labels        = labels[A_min:A_max]
        Gamma_labels    = labels[Gamma_min:Gamma_max]
        
        if not only_radial:
            if mixed_modes:
                
                V_nl     = flat_samples[:,V_nl_min:V_nl_max]
                V_nl_labels    = labels[V_nl_min:V_nl_max]
                
                if (V_l_min-V_l_max) == 0:
                    V_l     = flat_samples[:,V_l_min]
                    V_l_labels    = labels[V_l_min]
                else:
                    V_l     = flat_samples[:,V_l_min:V_l_max]
                    V_l_labels    = labels[V_l_min:V_l_max]
                
                if fit_split:
                    nu_s    = flat_samples[:,nu_s_min:nu_s_max]
                    i       = flat_samples[:,i_index]
                    #i = np.rad2deg(np.arccos(i))
                    
                    nu_s_labels = labels[nu_s_min:nu_s_max]
                    i_label     = labels[i_index]
            else:
                if (V_l_min-V_l_max) == 0:
                    V_l     = flat_samples[:,V_l_min]
                    V_l_labels    = labels[V_l_min]
                else:
                    V_l     = flat_samples[:,V_l_min:V_l_max]
                    V_l_labels    = labels[V_l_min:V_l_max]
                
                if fit_split:
                    nu_s    = flat_samples[:,nu_s_index]
                    i       = flat_samples[:,i_index]
                    #i = np.rad2deg(np.arccos(i))
                    
                    nu_s_labels = labels[nu_s_index]
                    i_label     = labels[i_index]
        

        if fit_bg:
            sigma_min, sigma_max, tau_min, tau_max, W_index = bg.get_index(bg_k)
            
            if (sigma_min-sigma_max) == 0:
                sigma_k = flat_samples[:,sigma_min]
                tau_k   = flat_samples[:,tau_min]
            else:
                sigma_k = flat_samples[:,sigma_min:sigma_max]
                tau_k   = flat_samples[:,tau_min:tau_max]
            W       = flat_samples[:,W_index]
            
            if (sigma_min-sigma_max) == 0:
                sigma_k_lab = labels[sigma_min]
                tau_k_lab   = labels[tau_min]
            else:
                sigma_k_lab = labels[sigma_min:sigma_max]
                tau_k_lab   = labels[tau_min:tau_max]
            W_lab       = labels[W_index]
            
            plot_params = np.c_[sigma_k, tau_k, W]
            labels_short = np.concatenate((sigma_k_lab, tau_k_lab, [W_lab]))
            
            # Make plot pretty:
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            f = corner.corner(np.array(plot_params), 
                              labels=labels_short,
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
        
        l1_indx = -1
        loop_values = zip(nu_nl.T, A_nl.T, Gamma_nl.T, nu_labels, A_labels, Gamma_labels)
        for nl, (nu, A, Gamma, nu_lab, A_lab, Gamma_lab) in enumerate(loop_values):
            plot_dim = 3
            plot_params = [nu.T, A.T, Gamma.T]
            labels_short = [nu_lab, A_lab, Gamma_lab]
            
            if not only_radial:
                current_l = modes[mask_fit][nl]['l']
                if current_l == 1:
                    l1_indx += 1
                
                if mixed_modes:
                    if current_l == 1:
                        plot_dim += 1
                        plot_params.append(V_nl[:,l1_indx])
                        labels_short.append(V_nl_labels[l1_indx])
                    
                if fit_split & (current_l == 1):
                    plot_dim += 2
                    
                    if mixed_modes:
                        plot_params.append(nu_s[:,l1_indx])
                        labels_short.append(nu_s_labels[l1_indx])
                    else:
                        plot_params.append(nu_s)
                        labels_short.append(nu_s_labels)
                    
                    plot_params.append(i)
                    labels_short.append(i_label)
                
                # V_l implementation
                if current_l == 0:
                    if fit_split | mixed_modes:
                        l_maxfit = 1
                    else:
                        l_maxfit = 0
                    
                    ls = np.unique(modes['l'])
                    plot_dim += len(ls[ls>l_maxfit])
                    for V, V_lab in zip(V_l.T, V_l_labels):
                        plot_params.append(V.T)
                        labels_short.append(V_lab)
            
            # Make plot pretty:
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            f = corner.corner(np.array(plot_params).T, 
                              labels=labels_short,
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
                bad_chars = ['$', '\\', '{', '}']
                for j in bad_chars:
                    nu_lab = nu_lab.replace(j, '')
                
                ax = plt.gca()
                ax.figure.savefig(res_dir+'/CornerPlot_{}.png'.format(nu_lab), dpi=300)
            
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
        np.savetxt(res_dir+'/BestFitResults.txt', results, delimiter='\t', fmt='%s')
    
    return best_theta


def PlotBestFit(ax, nu, pd, target, modes, bg_k, res_dir=None, BGax=None, 
                discard_num=0, sampler=None, best_theta=None, 
                chain_plot=True, corner_plot=True,
                only_radial=False, fit_split=True, mixed_modes=True,
                fit_bg=True, BG_bestFit=None, alt2=False):
    
    
    if sampler is None and best_theta is None:
        print('Error. Either sampler or best_theta has to be specified')
        return None
    
    if sampler is not None:
        best_theta = cornerPlot(sampler, nu, modes, bg_k, discard_num, res_dir, 
                                chain=chain_plot, plot=corner_plot, 
                                only_radial=only_radial, 
                                fit_split=fit_split, mixed_modes=mixed_modes,
                                fit_bg=fit_bg)
    
    InputCatalog, ID, name, _, _, _, _, _ = myF.get_target_info(target)
    
    #### BACKGROUND ####
    sigma_min, sigma_max, tau_min, tau_max, W_index = bg.get_index(bg_k)
        
    if fit_bg and BGax is not None:
        sigma_k = best_theta[sigma_min:sigma_max]
        tau_k   = best_theta[tau_min:tau_max]
        W       = best_theta[W_index]
        
        N_nu, N_comps = bg.bg_model(nu, sigma_k, tau_k, components=True)
        bg_fit = N_nu+W
        
        styles = ['dotted', 'dashdot', 'solid']
        for i, N_comp in enumerate(N_comps):
            ii = i
            if (bg_k == 3) | alt2:
                ii -= 1
            BGax.loglog(nu, N_comp, linestyle=styles[ii], color='lime', 
                      label=str(ii+1)+r'. comp. of $\mathcal{B}$')
        BGax.hlines(W, nu.min(), nu.max(), linestyle='dashed', color='lime', 
                  label=r'$W$')
        BGax.loglog(nu, bg_fit, 'r', label=r'Total $\mathcal{B}$')
        BGax.legend(loc='upper left', bbox_to_anchor=(1,1))
        # BGax.figure.suptitle('Background fit for {} {} ({})'.format(InputCatalog, ID, name))
        
        if res_dir is not None:
            BGax.figure.savefig(res_dir+'/BG_bestFit.png', dpi=300)
    
    
    #### MODES ####    
    Nmodes = len(modes)
    best_nu = best_theta[:Nmodes]
    ls = np.unique(modes['l'])
    c = myF.plotting_colors()
    m = myF.get_markers()

    P_fit = ps.model(best_theta, nu, InputCatalog, modes, bg_k, 
                     only_radial=only_radial, 
                     fit_split=fit_split, mixed_modes=mixed_modes,
                     fit_bg=fit_bg, BG_bestFit=BG_bestFit)
    
    ax.plot(nu, P_fit, 'orange', label='Power spectrum fit\n')
    
    F_like = False
    if not F_like:
        extra = [1.18, 1.08, 0.98, 0.88]
        for l in ls:
            l_arr = best_nu[modes['l']==l]
            ax.scatter(l_arr, np.ones(len(l_arr))*(max(pd)*extra[l]), 
                       s=80, zorder=10, marker=m[l], color=c[l], 
                       edgecolors='k', label=r'$\ell={}$'.format(l), alpha=.8)
            ax.set_ylim([0,max(pd)*1.27])
                
    ax.figure.subplots_adjust(top=0.9,
                              bottom=0.15,
                              left=0.1,
                              right=0.77,
                              hspace=0.5,
                              wspace=0.2)
    
    ax.set_xlim([nu.min(), nu.max()])

    #handles, labels = ax.get_legend_handles_labels()
    #sorted_handles = list(np.append(np.append(handles[0:2],handles[-1]),handles[2:-1]))
    
    #ax.legend(handles=sorted_handles, loc='upper left', bbox_to_anchor=(1,1))
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    #ax.figure.suptitle('{} {} ({})'.format(InputCatalog, ID, name))
    ax.text(1, 1, name, transform=ax.transAxes, 
            va='bottom', ha='right', fontsize=10)
    
    if res_dir is not None:
        ax.figure.savefig(res_dir+'/PDS_bestFit.png', dpi=300)
    
    # plot interpolated modes together with best results
    A_min, A_max, Gamma_min, Gamma_max = ps.get_index('fit', modes,  
                                                   fit_split=fit_split, 
                                                   mixed_modes=mixed_modes)
    nu_nl    = best_theta[:len(modes)]
    A_nl     = best_theta[A_min:A_max]
    Gamma_nl = best_theta[Gamma_min:Gamma_max]
    
    if fit_split | mixed_modes:
        l_min = 2
    else:
        l_min = 1
    
    ps.interpolate_modes(modes, nu_nl, A_nl, Gamma_nl, l_min, plot=True)#, res_dir=res_dir)
    
    if res_dir is not None:
        ax = plt.gca()
        ax.text(1, 1, name, transform=ax.transAxes,
            va='bottom', ha='right', fontsize=10)
        ax.figure.savefig(res_dir+'/A_Gamma_bestFit.png', dpi=300)
    
    return None

##########  .  ##########
