A peak-bagging software developed for my Master's thesis in Astronomy, to extract oscillation mode parameters from the power spectra of six solar-like oscillators in the subgiant evolutionary state.

Following a Bayesian approach, the software exploits the affine invariant ensemble MCMC sampler, emcee (see https://emcee.readthedocs.io/en/stable/). 

The software comes in two steps:
  1) A background fit to entire power spectrum
  2) A power spectrum fit centred on a local region encapsulating the oscillation power envelope. This includes fitting a sum of Lorentzians profiles to the oscillation modes

Priors are found in 'MyFunctions/'

The software was executed on the Linux cluster Grendel, facilitated by the Centre for Scientific Computing, Aarhus (CSCAA) (see https://phys.au.dk/forskning/faciliteter/cscaa/).
