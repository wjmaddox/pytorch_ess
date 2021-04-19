import torch

from .elliptical_slice import EllipticalSliceSampler


class MeanEllipticalSliceSampler(EllipticalSliceSampler):
    def __init__(self, f_init, dist, lnpdf, nsamples, pdf_params=()):
        """
        Implementation of elliptical slice sampling (Murray, Adams, & Mckay, 2010).
        f_init: initial value of `f`
        dist: multivariate normal to sample from to sample from
        lnpdf: likelihood function
        n_samples: number of samples
        pdf_params: callable arguments for lnpdf
        """
        mean_vector = dist.mean

        demeaned_lnpdf = lambda g: lnpdf(g + mean_vector, *pdf_params)

        demeaned_init = f_init - mean_vector

        samples = dist.sample(sample_shape = torch.Size((nsamples,))).transpose(-1, -2)
        demeaned_samples = samples - mean_vector.unsqueeze(1)

        super(MeanEllipticalSliceSampler, self).__init__(demeaned_init, demeaned_samples, demeaned_lnpdf, nsamples, pdf_params=())

        self.mean_vector = mean_vector

    def run(self):
        self.f_sampled, self.ell = super().run()

        #add means back into f_sampled
        self.f_sampled = self.f_sampled + self.mean_vector.unsqueeze(1)

        return self.f_sampled, self.ell