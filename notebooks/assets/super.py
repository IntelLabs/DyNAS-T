import torch.nn as nn
import torch.nn.functional as F


def _sample_weight(weight, sample_in_dim, sample_out_dim):

    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]

    return sample_weight


def _sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias


class SuperLinear(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # Define SuperNetwork Bounds
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}
        super().reset_parameters()

        self.profiling = False

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = _sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = _sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias'])
