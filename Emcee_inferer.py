from typing import Iterable, Union
import time
import numpy as np
import emcee
from tqdm import tqdm
from utils.parallel import parallel_methods

class inferer:

    def __init__(self, *args, **kwargs):
        self.priors = None

    def log_likelihood(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def log_prior(self, params: np.ndarray) -> float:
        raise NotImplementedError

# This implements the emcee library for Ensemble Monte Carlo method
    def log_probability(self, params: np.ndarray):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)

    def Ensembleinfer(self,nwalkers:int,niter: int):
        initial = np.array([prior.sample() for prior in self.priors])
        ndim = len(initial)
        p0_0 = [initial + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
        p0 = p0_0
        L_p = self.log_probability
        L_like = self.log_likelihood
        sampler = emcee.EnsembleSampler(nwalkers, ndim, L_p)
        print("Running sampler: ")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
        return sampler, pos, prob, state, L_like
