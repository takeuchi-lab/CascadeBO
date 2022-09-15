from .model_gpytorch import (SingleTaskGP, IndependentMOGP, CascadeMOGP, fit_param, MultitaskGP)

__all__ = [
    'fit_param',
    'SingleTaskGP',
    'IndependentMOGP',
    'CascadeMOGP',
    'MultitaskGP'
]
