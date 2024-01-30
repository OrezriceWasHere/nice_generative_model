"""NICE model
"""
import typing

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np
"""Additive coupling layer.
"""

odd, even = list(), list()

def split_x1_x2(x, mask_config) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Split input x into two parts along channel dimension.

    Args:
        x: input tensor.
        mask_config: 1 if transform odd units, 0 if transform even units.
    Returns:
        two tensors.
    """
    n = x.shape[1]
    assert n % 2 == 0, "Cannot split to exact half at odd length"
    global odd, even
    if not odd:
        even, odd = list(range(0, n, 2)), list(range(1, n, 2))
    if mask_config == 1:
        return x[:, odd], x[:, even]
    else:
        return x[:, even], x[:, odd]

def merge_x1_x2(x1,x2, mask_config) -> torch.Tensor:
    """Split input x into two parts along channel dimension.

    Args:
        x: input tensor.
        mask_config: 1 if transform odd units, 0 if transform even units.
    Returns:
        two tensors.
    """
    x = torch.zeros((x1.shape[0], x1.shape[1] * 2))
    global odd, even
    if mask_config == 1:
        x[:,odd] += x1
        x[:, even] += x2
    else:
        x[:, even] += x1
        x[:, odd] += x2

    return x

class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        #TODO fill in

        # Input layer
        layers = [
            nn.Linear(in_out_dim // 2, mid_dim),
            nn.LeakyReLU()
        ]

        # Middle layers
        for _ in range(hidden):
            layers += [
                nn.Linear(mid_dim, mid_dim),
                nn.LeakyReLU()
            ]

        # Output layer
        layers += [
            nn.Linear(mid_dim, in_out_dim // 2)
        ]

        self.coupling_model = nn.Sequential(*layers)
        self.mask_config = mask_config


    def forward(self, x, log_det_J, compute_backwards=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            compute_backwards: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """

        #TODO fill in

        if not compute_backwards:
            x1, x2 = split_x1_x2(x, self.mask_config)
            y1, y2 = x1, x2 + self.coupling_model(x1)
            y = merge_x1_x2(y1, y2, self.mask_config)
            return y, log_det_J

        else:
            y1, y2 = split_x1_x2(x, self.mask_config)
            x1, x2 = y1, y2 - self.coupling_model(y1)
            x = merge_x1_x2(x1, x2, self.mask_config)
            return x, log_det_J

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        self.in_out_dim = in_out_dim
        self.mask_config = mask_config
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.t_transform = nn.Conv2d(in_out_dim//2, in_out_dim//2, kernel_size=3, padding=1)
        # self.s_transform = nn.Sequential(
        #     nn.Conv2d(in_out_dim//2, in_out_dim//2, kernel_size=3, padding=1),
        #     torch.nn.modules.activation.())
        # )
        #TODO fill in

    def forward(self, x, log_det_J, compute_backwards=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            compute_backwards: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """

        x1, x2 = x[:, :self.in_out_dim//2], x[:, self.in_out_dim//2:]
        x_same, x_transform = x1, x2
        if self.mask_config == 1:
            x_same, x_transform = x2, x1

        y1 = x_same
        for i in range(self.hidden):
            y1 = self.leaky_relu(
                nn.Linear(self.in_out_dim//2, self.mid_dim)(y1))
        shift = nn.Linear(self.mid_dim, self.in_out_dim//2)(y1)
        y2 = x_transform + shift
        if compute_backwards:
            y1, y2 = y2, y1
            shift = -shift
        x = torch.cat([y1, y2], dim=1)
        log_det_J += torch.sum(shift, dim=1)
        return x, log_det_J

        #TODO fill in

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        # self.scale = nn.Parameter(
        #     torch.zeros((1, dim)), requires_grad=True)
        self.scale = nn.Parameter(torch.zeros((1, dim), requires_grad=True))
        self.eps = 1e-5

    def forward(self, x, log_det_J, compute_backwards=False):
        """Forward pass.

        Args:
            x: input tensor.
            compute_backwards: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale) + self.eps
        #TODO fill in
        # We only sum instead of applying log because scaling is already in log space
        this_log_det_J = torch.sum(self.scale).item()

        if compute_backwards:
            return x / scale, log_det_J - this_log_det_J

        return x * scale, log_det_J + this_log_det_J





"""Standard logistic distribution.
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,
        in_out_dim, mid_dim, hidden,device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output x
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type

        #TODO fill in

        layers = [AdditiveCoupling(in_out_dim=in_out_dim,
                                   mid_dim=mid_dim,
                                   hidden=hidden,
                                   mask_config=i % 2)
                  for i in range(coupling)]

        layers += [Scaling(in_out_dim)]

        self.layers = nn.ModuleList(layers)

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        #TODO fill in
        log_det_J = 0
        for layer in reversed(self.layers):
            z, log_det_J = layer(z, log_det_J, compute_backwards=True)

        x = z
        return x, log_det_J



    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        #TODO fill in
        log_det_J = 0
        for layer in self.layers:
            x, log_det_J = layer(x, log_det_J)

        z = x
        return z, log_det_J

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256) * self.in_out_dim #log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        #TODO

        x, log_det_J = self.f_inverse(z)
        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.f_inverse(x)
