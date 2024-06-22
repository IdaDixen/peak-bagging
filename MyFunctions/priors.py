#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:03:46 2024

@author: idadixenskaarupwolm
"""

import numpy as np
import math


##########  Priors  ##########
def uniform_prior(x, xmin, xmax):
    
    if xmin < x < xmax:
        return np.log(1/(xmax-xmin))
    
    return -np.inf

def Jeffreys_prior(x, xmin, xmax):
    
    if xmin < x < xmax:
        return np.log(1/(x*np.log(xmax/xmin)))
    
    return -np.inf


def mod_Jeffreys_prior(x, xmin, xmax, xuni):
    
    if xmin < x < xmax:
        return np.log(1 / ( (x + xuni) * np.log((xuni+xmax) / xuni)) )
    
    return -np.inf


def truncated_Gaussian_prior(x, xmin, xmax, x0, sigma):
    
    if xmin < x < xmax:
        D = ( math.erf((xmax-x0) / (np.sqrt(2)*sigma)) - math.erf((xmin-x0) / (np.sqrt(2)*sigma)) ) / 2
        return np.log( 1/(D*np.sqrt(2*np.pi)*sigma) * np.exp(-(x-x0)**2 / (2*sigma**2)) )
    
    return -np.inf