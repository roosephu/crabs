import torch.distributions as distributions
import torch.distributions.transforms as transforms


class TanhGaussian(distributions.TransformedDistribution):
    has_rsample = True

    def __init__(self, loc, scale):
        base_dist = distributions.Normal(loc, scale)
        super().__init__(base_dist, transforms.TanhTransform(cache_size=1))

    # approx?
    @property
    def mean(self):
        return self.base_dist.mean.tanh()

    # approx?
    @property
    def stddev(self):
        return self.base_dist.stddev


__all__ = ['TanhGaussian']
