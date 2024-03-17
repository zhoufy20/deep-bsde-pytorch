#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :equation.py
# @Time      :2024/3/12 16:22
# @Author    :Feiyu
# @Main      ：This file contains the definition of the partial differential equation
# and any necessary boundary conditions.

import torch
import numpy as np
from scipy.stats import multivariate_normal as normal
import default_parameters

class Equation:
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        self._dim = eqn_config['dim']
        self._total_time = eqn_config['total_time']
        self._num_time_interval = eqn_config['num_time_interval']
        self._delta_t = (self.eqn_total_time + 0.0) / self.eqn_num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_th(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_th(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @ property
    def eqn_dim(self):
        return self._dim

    @ property
    def eqn_num_time_interval(self):
        return self._num_time_interval

    @property
    def eqn_total_time(self):
        return self._total_time

    @property
    def eqn_sqrt_delta_t(self):
        return self._sqrt_delta_t

    @property
    def eqn_delta_t(self):
        return self._delta_t


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(AllenCahn, self).__init__(eqn_config)
        self.x_init = np.zeros(self.eqn_dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,self.eqn_dim,
                                     self.eqn_num_time_interval]) * self.eqn_sqrt_delta_t
        x_sample = np.zeros([num_sample, self.eqn_dim, self.eqn_num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.eqn_dim]) * self.x_init
        for i in range(self.eqn_num_time_interval):
            # the Euler-Maruyama scheme to simulate the paths of X for each time step
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        """Generator function in the PDE."""
        return y - torch.pow(y, 3)

    def g_th(self, t, x):
        """Terminal condition of the PDE."""
        return 0.5 / (1 + 0.2 * torch.sum(x**2, dim=1, keepdim=True))

class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.eqn_dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.eqn_dim, self.eqn_num_time_interval]) * self.eqn_sqrt_delta_t
        x_sample = np.zeros([num_sample, self.eqn_dim, self.eqn_num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.eqn_dim]) * self.x_init
        for i in range(self.eqn_num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        return -self.lambd * torch.sum(z ** 2, dim=1, keepdim=True)

    def g_th(self, t, x):
        return torch.log((1 + torch.sum(x ** 2, dim=1, keepdim=True))/2)


class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """
    def __init__(self, eqn_config):
        super(PricingDefaultRisk, self).__init__(eqn_config)
        self.x_init = np.ones(self.eqn_dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02   # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.eqn_dim, self.eqn_num_time_interval]) * self.eqn_sqrt_delta_t
        x_sample = np.zeros([num_sample, self.eqn_dim, self.eqn_num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.eqn_dim]) * self.x_init
        for i in range(self.eqn_num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.eqn_delta_t) * x_sample[:, :, i] + (
                self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        piecewise_linear = torch.nn.functional.relu(
            torch.nn.functional.relu(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_th(self, t, x):
        min_values, _ = torch.min(x, dim=1, keepdim=True)
        return min_values

class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.eqn_dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.eqn_dim

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.eqn_dim, self.eqn_num_time_interval]) * self.eqn_sqrt_delta_t
        x_sample = np.zeros([num_sample, self.eqn_dim, self.eqn_num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.eqn_dim]) * self.x_init
        factor = np.exp((self.mu_bar-(self.sigma**2)/2)*self.eqn_delta_t)
        for i in range(self.eqn_num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        # temp = torch.sum(z, dim=1, keepdim=True) / self.sigma
        # return -self.rl * y - (self.mu_bar - self.rl) * temp + (
        #         (self.rb - self.rl) * torch.maximum(temp - y, torch.zeros_like(temp)))
        temp = torch.sum(z, dim=1, keepdim=True) / self.sigma
        return -self.rl * y - (self.mu_bar - self.rl) * temp + ((self.rb - self.rl) * torch.clamp(temp - y, max=0))
    def g_th(self, t, x):
        # temp, _ = torch.max(x, dim=1, keepdim=True)
        # print(temp.size())
        # return torch.maximum(temp - 120, torch.zeros_like(temp)) - 2 * torch.maximum(temp - 150, torch.zeros_like(temp))
        temp = torch.max(x, dim=1, keepdim=True)[0]  # 获取最大值
        return torch.clamp(temp - 120, min=0) - 2 * torch.clamp(temp - 150, min=0)

class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(BurgersType, self).__init__(eqn_config)
        self.x_init = np.zeros(self.eqn_dim)
        self.y_init = 1 - 1.0 / (1 + np.exp(0 + np.sum(self.x_init) / self.eqn_dim))
        self.sigma = self.eqn_dim + 0.0

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.eqn_dim, self.eqn_num_time_interval]) * self.eqn_sqrt_delta_t
        x_sample = np.zeros([num_sample, self.eqn_dim, self.eqn_num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.eqn_dim]) * self.x_init
        for i in range(self.eqn_num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        return (y - (2 + self.eqn_dim) / 2.0 / self.eqn_dim) * torch.sum(z, dim=1, keepdim=True)

    def g_th(self, t, x):
        return 1 - 1.0 / (1 + torch.exp(t + torch.sum(x, dim=1, keepdim=True) / self.eqn_dim))


class QuadraticGradient(Equation):
    """
    An example PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(QuadraticGradient, self).__init__(eqn_config)
        self.alpha = 0.4
        self.x_init = np.zeros(self.eqn_dim)
        base = self.eqn_total_time + np.sum(np.square(self.x_init) / self.eqn_dim)
        self.y_init = np.sin(np.power(base, self.alpha))
        print(self.y_init)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.eqn_dim, self.eqn_num_time_interval]) * self.eqn_sqrt_delta_t
        x_sample = np.zeros([num_sample, self.eqn_dim, self.eqn_num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.eqn_dim]) * self.x_init
        for i in range(self.eqn_num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        x_square = torch.sum(x**2, dim=1, keepdim=True)
        base = self.eqn_total_time - t + x_square / self.eqn_dim
        base_alpha = torch.pow(base, self.alpha)
        derivative = self.alpha * torch.pow(base, self.alpha - 1) * torch.cos(base_alpha)
        term1 = torch.sum(torch.square(z), dim=1, keepdim=True)
        term2 = -4.0 * (derivative ** 2) * x_square / (self.eqn_dim ** 2)
        term3 = derivative
        term4 = -0.5 * (2.0 * derivative + 4.0 / (self.eqn_dim ** 2) * x_square * self.alpha * ((self.alpha - 1)
                * torch.pow(base, self.alpha - 2) * torch.cos(base_alpha) -
                (self.alpha * torch.pow(base, 2 * self.alpha - 2) * torch.sin(base_alpha))))
        return term1 + term2 + term3 + term4

    def g_th(self, t, x):
        return torch.sin(torch.pow(torch.sum(x**2, dim=1, keepdim=True) / self.eqn_dim, self.alpha))


class ReactionDiffusion(Equation):
    """
    Time-dependent reaction-diffusion-type example PDE in Section 4.7 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(ReactionDiffusion, self).__init__(eqn_config)
        self.TH_DTYPE = eqn_config['default_Config']['TH_DTYPE']
        self.kappa = 0.6
        self.lambd = torch.tensor(1 / np.sqrt(self.eqn_dim), dtype=self.TH_DTYPE)
        self.x_init = np.zeros(self.eqn_dim)
        self.y_init = 1 + self.kappa + np.sin(self.lambd * np.sum(self.x_init)) * np.exp(
            -self.lambd * self.lambd * self.eqn_dim * self.eqn_total_time / 2)
        print(self.y_init)
    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.eqn_dim, self.eqn_num_time_interval]) * self.eqn_sqrt_delta_t
        x_sample = np.zeros([num_sample, self.eqn_dim, self.eqn_num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.eqn_dim]) * self.x_init
        for i in range(self.eqn_num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        exp_term = torch.exp((self.lambd ** 2) * self.eqn_dim * (t - self.eqn_total_time) / 2)
        sin_term = torch.sin(self.lambd * torch.sum(x, 1, keepdims=True))
        temp = y - self.kappa - 1 - sin_term * exp_term
        return torch.min(torch.tensor(1.0, dtype=torch.float64), torch.square(temp))


    def g_th(self, t, x):
        return 1 + self.kappa + torch.sin(self.lambd * torch.sum(x, dim=1, keepdim=True))


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    allen = AllenCahn(default_parameters.AllenCahnConfig)
    dw, x = allen.sample(20)

    # hjb = HJBLQ(default_parameters.HJBConfig)
    # print(allen.eqn_num_time_interval)
    dw,x = allen.sample(20)

