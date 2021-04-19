from typing import Tuple

import math
import torch

class LinConEllipticalSliceSampler:
    def __init__(self, A, b, n_samples, f_init=None):
        r"""
        A: M x d tensor
        b: M tensor
        n_samples: number of samples
        f_init: initial value

        samples from linearly constrained gaussian subject to constraints A x + b > =0
        implementation is a pytorch version of https://github.com/alpiges/LinConGauss
        """

        if f_init is None:
            found = False
            while not found:
                f_init = torch.randn(A.shape[-1], 1)
                found = ((A.matmul(f_init) + b) >= 0).sum() == A.shape[0]
        self.f_init = f_init
        self.A = A
        self.b = b
        self.n_samples = n_samples

    def find_rotated_intersections(self, x0, x1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates the interactions by the first angle in self.slices (rotation angle)
        and makes sure that all angles lie in [0, 2*pi]
        :return: rotation angle (np.float), shifted angles (np.ndarray)
        """
        slices = self.find_active_intersections(x0, x1)
        rotation_angle = slices[0]
        slices = slices - rotation_angle

        return rotation_angle, slices + (slices < 0)*2.*math.pi

    def _compute_active_difference(self, theta, x0, x1, delta_theta=1e-4):
        pos_diff = (self.A.matmul(self._sample_on_slice(theta + delta_theta, x0, x1)) + self.b >= 0).sum()
        neg_diff = (self.A.matmul(self._sample_on_slice(theta - delta_theta, x0, x1)) + self.b >= 0).sum()
        return (pos_diff == self.A.shape[0]).float() - (neg_diff == self.A.shape[0]).float()

    def _find_active_directions(self, theta, x0, x1, delta_theta=1e-4):
        return torch.stack([self._compute_active_difference(t, x0, x1, delta_theta) for t in theta])

    def find_active_intersections(self, x0, x1) -> torch.Tensor:
        """
        Find angles of those intersections that are at the boundary of the integration domain
        by adding and subtracting a small angle and evaluating on the ellipse to see if we are on the boundary of the
        integration domain.
        :return: angles of active intersection in order of increasing angle theta such that activation happens in
        positive direction. If a slice crosses theta=0, the first angle is appended at the end of the array.
        Every row of the returned array defines a slice for elliptical slice sampling.
        """
        delta_theta = 1.e-5 * 2.*math.pi
        theta = self.intersection_angles(x0, x1)

        active_directions = self._find_active_directions(theta, x0, x1, delta_theta)
        theta_active = theta[active_directions!=0]

        while theta_active.shape[-1] % 2 == 1:
            # Almost tangential ellipses, reduce delta_theta
            delta_theta = 1.e-1 * delta_theta
            active_directions = self._find_active_directions(theta, x0, x1, delta_theta)
            theta_active = theta[active_directions!=0]

        nonzero_active_directions = active_directions[active_directions!=0]
        if nonzero_active_directions.shape != torch.Size([0]) and nonzero_active_directions[0] == -1:
            theta_active = torch.cat((theta_active[1:].view(-1), theta_active[0].view(-1)))

        return theta_active

    def intersection_angles(self, x0, x1) -> torch.Tensor:
        """ Compute all of the up to 2M intersections of the ellipse and the linear constraints """
        g1 = self.A.matmul(x0)
        g2 = self.A.matmul(x1)

        r = torch.sqrt(g1**2 + g2**2)
        phi = 2*torch.atan(g2/(r+g1)).squeeze()

        # two solutions per linear constraint, shape of theta: (M, 2)
        arg = - (self.b / r.squeeze(-1)).squeeze()
        theta = torch.zeros((self.A.shape[0], 2), dtype=self.A.dtype, device=self.A.device)

        # write NaNs if there is no intersection
        arg[torch.absolute(arg) > 1] = torch.tensor(float("nan"))
        theta[:, 0] = torch.arccos(arg) + phi
        theta[:, 1] = - torch.arccos(arg) + phi
        theta = theta[torch.isfinite(theta)]

        return torch.sort(theta + (theta < 0.)*2.*math.pi)[0]   # in [0, 2*pi]

    def _sample_on_slice(self, angle, x0, x1) -> torch.Tensor:
        return x0 * torch.cos(angle) + x1 * torch.sin(angle)

    def elliptical_slice(
        self, 
        initial_theta: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = torch.randn_like(initial_theta)

        # we need to draw the new angle t_new
        # start by finding the slices
        rotation_angle, rotation_slices = self.find_rotated_intersections(initial_theta, x1)

        rotation_slices = rotation_slices.reshape(-1, 2)
        rotation_lengths = rotation_slices[:, 1] - rotation_slices[:, 0]

        # now construct the rotation angle
        cum_lengths = torch.cat((torch.zeros(1, device = rotation_lengths.device, dtype=rotation_lengths.dtype), 
            torch.cumsum(rotation_lengths)))
        random_angle = torch.rand(1) * cum_lengths[-1]
        idx = torch.searchsorted(cum_lengths, random_angle) - 1
        t_new = rotation_slices[idx, 0] + random_angle - cum_lengths[idx] + rotation_angle

        return self._sample_on_slice(t_new, initial_theta, x1)

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.f_sampled = torch.zeros(self.A.shape[-1], self.n_samples, device=self.f_init.device, dtype=self.f_init.dtype)

        f_cur = self.f_init
        for ii in range(self.n_samples):
            # TODO: check if this will cause autocorrelation?
            if ii > 0:
                f_cur = self.f_sampled[:,ii-1]

            self.f_sampled[:, ii] = self.elliptical_slice(f_cur)

        # no reasn to return log prob
        return self.f_sampled, None

