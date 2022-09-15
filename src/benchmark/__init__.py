from .benchmarks import (SynthFunction, GPTest, NegAckley, NegRosenbrock, NegLevy, NegSchwefel, NegBeale,
                        NegRastrigin, NegSphere, NegMatyas,
                         Constant,
                         set_n_cpu, RealFunction,
                         PDiffusion, PDiffusion1st, PDiffusion2nd,
                         PC1D, PC1DEmu,
                         EC_1st, EC_2nd, Bookmark)

__all__ = [
    'set_n_cpu',
    'SynthFunction', 'GPTest', 'NegAckley', 'NegRosenbrock', 'NegLevy', 'NegSchwefel', 'NegBeale', 'NegRastrigin', 'NegSphere', 'NegMatyas', 'Constant',
    'RealFunction', 'PDiffusion1st', 'PDiffusion2nd', 'PDiffusion', 'PC1DEmu', 'PC1D', 'EC_1st', 'EC_2nd', 'Bookmark'
]
