#####################################
# Elliptical slice sampling
# Implementation comes from https://github.com/jobovy/bovy_mcmc/blob/master/bovy_mcmc/elliptical_slice.py
# This version is copied over from the original pytorch implementation at: https://github.com/wjmaddox/spectralgp/
#####################################

from typing import Tuple

import torch
import math
import numpy as np

class EllipticalSliceSampler:
    def __init__(self, f_init, f_priors, lnpdf, n_samples, pdf_params=()):
        r"""
        Implementation of elliptical slice sampling (Murray, Adams, & Mckay, 2010).
        f_init: initial value of `f`
        f_priors: prior samples from N(0, \Sigma)
        lnpdf: likelihood function
        n_samples: number of samples
        pdf_params: callable arguments for lnpdf
        """

        self.n = f_init.nelement()
        self.f_init = f_init
        self.f_priors = f_priors
        self.lnpdf = lnpdf
        self.n_samples = n_samples
        self.pdf_params = pdf_params

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.f_sampled = torch.zeros(self.n, self.n_samples, device=self.f_init.device, dtype=self.f_init.dtype)
        self.ell = torch.zeros(self.n_samples, 1, device=self.f_init.device, dtype=self.f_init.dtype)

        f_cur = self.f_init
        for ii in range(self.n_samples):
            if ii == 0:
                ell_cur = self.lnpdf(f_cur, *self.pdf_params)
            else:
                f_cur = self.f_sampled[:,ii-1]
                ell_cur = self.ell[ii-1, 0]

            next_f_prior = self.f_priors[:,ii]

            self.f_sampled[:, ii], self.ell[ii] = self.elliptical_slice(f_cur, next_f_prior, 
                cur_lnpdf=ell_cur, pdf_params=self.pdf_params)

        return self.f_sampled, self.ell

    def elliptical_slice(
        self, 
        initial_theta: torch.Tensor, 
        prior: torch.Tensor,
        pdf_params: Tuple =(), 
        cur_lnpdf: torch.Tensor=None, 
        angle_range: float=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        D= len(initial_theta)
        if cur_lnpdf is None:
            cur_lnpdf= self.lnpdf(initial_theta, *pdf_params)

        ## FORCING THE RIGHT PRIOR TO BE GIVEN ##
        # Set up the ellipse and the slice threshold
        if len(prior.shape) == 1: #prior = prior sample
            nu = prior
        else: #prior = cholesky decomp
            if not prior.shape[0] == D or not prior.shape[1] == D:
                raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
            nu = prior.transpose(-1, -2).matmul(torch.randn(prior.shape[:-2], 1, device = prior.device, dtype=prior.dtype))

        hh = torch.rand(1).log() + cur_lnpdf

        # Set up a bracket of angles and pick a first proposal.
        # "phi = (theta'-theta)" is a change in angle.
        if angle_range is None or angle_range == 0.:
            # Bracket whole ellipse with both edges at first proposed point
            phi= torch.rand(1)*2.*math.pi
            phi_min= phi-2.*math.pi
            phi_max= phi
        else:
            # Randomly center bracket on current point
            phi_min= -angle_range*torch.rand(1)
            phi_max= phi_min + angle_range
            phi= torch.rand(1)*(phi_max-phi_min)+phi_min

        # Slice sampling loop
        while True:
            # Compute xx for proposed angle difference and check if it's on the slice
            xx_prop = initial_theta*math.cos(phi) + nu*math.sin(phi)

            cur_lnpdf = self.lnpdf(xx_prop, *pdf_params)
            if cur_lnpdf > hh:
                # New point is on slice, ** EXIT LOOP **
                break
            # Shrink slice to rejected point
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
            # Propose new angle difference
            phi = torch.rand(1)*(phi_max - phi_min) + phi_min

        return (xx_prop,cur_lnpdf)
