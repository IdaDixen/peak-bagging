#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:14:44 2023



@author: Ida Dixen Skaarup Wölm




###############################################################################

                  Background fit of solar-like oscillators                     

###############################################################################


PREREQUISITS

Before run, make sure the following is changed to fit the run:
    -> target_indx
    -> test_info
    -> k (if not on Grendel)
If another filetype is needed, change pdsfile

Other settingd to consider:
    -> step_num
    -> discard_num
    -> mk_dir
    -> mCPU

Target info must be contained in: myF.get_target_info(target)
Morover the following files are needed:
    -> 'theta_init.csv' including uncertainties to make Gaussian ball to 
        radnomize walker positions
    -> 'priors.csv'
    
If the model is changed, remember to change priors, theta_init and unc as well



"""



import numpy as np
from astropy.io import ascii
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('MyFunctions')
import functions as myF
import MCMC_bg as mcmc

from datetime import datetime
import os
import time
start = time.time()




#####  Initial settings  #####

step_num = 2
discard_num = 0
k = 1 #int(sys.argv[2])
mk_dir = True
mCPU = False#True

target_list = myF.get_target_list()
target_indx = 0 #int(sys.argv[1])
target = target_list[target_indx]
InputCatalog, ID, name, fmin, fmax, _, _, _ = myF.get_target_info(target)
test_info = '1 component, long run, priors editted'



# make res dir if defined:
if mk_dir:
    
    res_dir = 'BG_results_'+InputCatalog+ID+'_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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





#####  Initial guesses  #####

print('Defining intial parameter guess')

Gauss_guess = ascii.read('MyFunctions/theta_init/gauss_P.csv')
numax = Gauss_guess['numax'][target_indx]
Amax = Gauss_guess['Amax'][target_indx]
# sigma_env = numax/(4*np.sqrt(2*np.log(2)))
sigma_env = Gauss_guess['sigma_env'][target_indx]

Gauss_unc = ascii.read('MyFunctions/uncertainties/gauss_P.csv')
numax_unc = Gauss_unc['numax'][target_indx]
Amax_unc = Gauss_unc['Amax'][target_indx]
sigma_env_unc = Gauss_unc['sigma_env'][target_indx]

theta_init = [numax, Amax, sigma_env]
unc = [numax_unc, Amax_unc, sigma_env_unc]


sigma_rms = 3382*np.array(numax)**(-0.6)
b_1 = 0.1*np.array(numax)
b_2 = 0.3*np.array(numax)
b_3 = np.array(numax)
tau1 = 1/(2*np.pi*b_1)
tau2 = 1/(2*np.pi*b_2)
tau3 = 1/(2*np.pi*b_3)
W = np.median(pd)

typical_unc = 0.05
sigma_rms_unc = typical_unc
tau1_unc = 1/(2*np.pi*b_1**2)*typical_unc
tau2_unc = 1/(2*np.pi*b_2**2)*typical_unc
tau3_unc = 1/(2*np.pi*b_3**2)*typical_unc
tau_unc = 5e-4
Wunc = 0.1*np.median(pd)


sigmas = sigma_rms*np.ones(3)
sigma_uncs = sigma_rms_unc*np.ones(3)

for j in range(k):
    theta_init.append(sigmas[j])
    unc.append(sigma_uncs[j])

taus = [tau1, tau2, tau3]
tau_uncs = [tau1_unc, tau2_unc, tau3_unc]

for j in range(k):
    if k != 3:
        j += 1
    theta_init.append(taus[j])
    unc.append(tau_uncs[j])

theta_init.append(W)
unc.append(Wunc)



#####  MCMC sampling #####

print('Starting sampling ({} parameters)\n\n'.format(len(theta_init)))

info_file = open(res_dir+'/run_info_BG_{}{}.txt'.format(InputCatalog, ID), 'w')
info_file.write('Target: {}{}\n'.format(InputCatalog, ID))
info_file.write('Test Info: {}\n\n'.format(test_info))
info_file.write('Dimensions (# of parameters): {}\n'.format(len(theta_init)))
# info_file.write('Walkers:  {}\n'.format(2*len(theta_init))) # now it is 500

sampler = mcmc.params_estimation(nu, pd, target, step_num, 
                                 theta_init=theta_init, unc=unc, 
                                 discard_num=discard_num, k=k, 
                                 res_dir=res_dir, multi_CPU=mCPU)



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
    print('Background fit was not initiated')
else:
    print('\n\nBackground fit has completed. Run time: {} {}.\n'.format(run_time, unit))

