import copy
import sys
from abc import ABC, abstractmethod
from typing import Union

import gpytorch
import gpytorch.utils.errors
import numpy as np
import torch
from botorch import fit_gpytorch_model
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, ProductKernel, RBFKernel, MaternKernel, MultitaskKernel, LCMKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.models import ExactGP, IndependentModelList






def cov_to_cor(covariance):
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    
    return correlation


def fit_param(mll, max_retries: int = 10):
    if isinstance(mll, SumMarginalLogLikelihood):
        original_state_dict = [copy.deepcopy(_mll.state_dict()) for _mll in mll.mlls]
    else:
        original_state_dict = copy.deepcopy(mll.state_dict())
    try:
        fit_gpytorch_model(mll, max_retries=max_retries)
    except gpytorch.utils.errors.NotPSDError:
        print(f'NotPSDError is raised in fit_param', file=sys.stderr)
        if isinstance(mll, SumMarginalLogLikelihood):
            for i in range(len(original_state_dict)):
                mll.mlls[i].load_state_dict(original_state_dict[i])
        else:
            mll.load_state_dict(original_state_dict)
        mll.eval()
    return


class SingleTaskGP(ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: Union[float, gpytorch.likelihoods.Likelihood],
                 outputscale_bounds, lengthscale_bounds, use_ard: bool = True, prior_mean: float = None):
        if isinstance(likelihood, float):
            noise_var = likelihood
            likelihood = GaussianLikelihood()
            likelihood.noise = noise_var
            likelihood.noise_covar.raw_noise.requires_grad = False
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        if prior_mean is None:
            prior_mean = train_y.mean()
        else:
            prior_mean = torch.tensor(prior_mean)
        self.mean_module.initialize(constant=prior_mean)
        self.mean_module.constant.requires_grad = False

        self.in_dim = train_x.shape[1]
        if outputscale_bounds is None:
            outputscale_bounds = (0.01, 250.)
        if lengthscale_bounds is None:
            if use_ard:
                lengthscale_bounds = [(0.1, 1000.) for _ in range(train_x.shape[1])]
            else:
                lengthscale_bounds = (0.1, 1000.)

        self.use_ard = use_ard
        if use_ard:
            in_dim = train_x.shape[1]
            kernels = []
            for i in range(in_dim):
                ls_bound = Interval(*lengthscale_bounds[i])
                kernels.append(RBFKernel(active_dims=(i,), lengthscale_constraint=ls_bound))
            self.rbf = ProductKernel(*kernels)
        else:
            if isinstance(lengthscale_bounds, list) and len(lengthscale_bounds) >= 1:
                lengthscale_bounds = lengthscale_bounds[0]
            self.rbf = RBFKernel(lengthscale_constraint=Interval(*lengthscale_bounds))

        self.covar_module = ScaleKernel(self.rbf, outputscale_constraint=Interval(*outputscale_bounds))
        return

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x: torch.Tensor, noiseless: bool = True):
        if x.dim() == 1:
            x = x.reshape(1, -1)
        pred = self(x)
        if noiseless:
            pred = self.likelihood(pred)
        return pred

    def add_observations(self, x: torch.Tensor, y: torch.Tensor):
        assert x.ndim == 2 and x.shape[1] == self.in_dim
        assert y.ndim == 1

        new_x = torch.cat([self.train_inputs, x])
        new_y = torch.cat([self.train_targets, y])
        self.set_train_data(new_x, new_y)
        return

    def set_lengthscale(self, value):
        self.rbf.lengthscale = value

    def set_outputscale(self, value):
        self.covar_module.outputscale = value


class MultiOutputGP(ABC):
    @abstractmethod
    def print_params(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def add_observations(self, x, y):
        pass


class IndependentMOGP(MultiOutputGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 outputscale_bounds=None, lengthscale_bounds=None,
                 noise_var=1e-4, use_ard: bool = True, prior_mean=None, lengthscale=None, outputscale=None):
        assert train_x.ndim == 2 and train_y.ndim == 2
        self.use_ard = use_ard
        self.in_dim = train_x.shape[1]
        self.out_dim = train_y.shape[1]
        self.outputscale_bounds = outputscale_bounds
        self.lengthscale_bounds = lengthscale_bounds
        self.noise_var = noise_var
        self.prior_mean = prior_mean
        each_train_y = [train_y[:, i] for i in range(self.out_dim)]
        each_model = []
        each_mll = []
        for i in range(self.out_dim):
            if outputscale_bounds is None:
                os_bound = None
            elif len(outputscale_bounds) == 1:
                os_bound = outputscale_bounds[0]
            else:
                os_bound = outputscale_bounds[i]

            model_i = SingleTaskGP(train_x=train_x, train_y=each_train_y[i],
                                   outputscale_bounds=os_bound,
                                   lengthscale_bounds=lengthscale_bounds,
                                   likelihood=noise_var, use_ard=use_ard, prior_mean=prior_mean)
            if lengthscale is not None:
                model_i.set_lengthscale(lengthscale)
            if outputscale is not None:
                model_i.set_outputscale(outputscale)

            mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
            model_i.eval()
            mll_i.eval()
            each_model.append(model_i)
            each_mll.append(mll_i)
        self.model = IndependentModelList(*each_model)
        self.mll = SumMarginalLogLikelihood(self.likelihood, self.model)
        self.model.eval()
        self.mll.eval()
        return

    @property
    def likelihood(self):
        return self.model.likelihood

    @property
    def train_inputs(self) -> torch.Tensor:
        return self.model.train_inputs[0][0]

    @property
    def train_targets(self) -> torch.Tensor:
        return torch.cat(self.model.train_targets).reshape(len(self.model.train_targets), -1).T

    def print_params(self, print_fixed_param: bool = False):
        i = 0
        for m, llh in zip(self.model.models, self.likelihood.likelihoods):
            print(f'model : {i}')
            if not print_fixed_param:
                print(f'constant_mean={m.mean_module.constant.item()}')
                print(f'noise_var={llh.noise.item()}')
            print(f'output_scale={m.covar_module.outputscale.item()}')
            if self.use_ard:
                ls = [float(k.lengthscale.detach()) for k in m.rbf.kernels]
            else:
                ls = m.rbf.lengthscale
            print(f'rbf:lengthscale={ls}')
            i += 1
        return

    def predict(self, x: torch.Tensor, noiseless: bool = True, output_i: int = None):
        rep_x = [x for _ in range(self.out_dim)]

        if self.model.training:
            self.model.eval()

        if output_i is not None:
            return self.model.models[output_i].predict(x, noiseless=noiseless)

        predictions = self.model(*rep_x)
        if not noiseless:
            predictions = self.likelihood(*predictions)
        mean = torch.stack([mvn.mean for mvn in predictions], -1)
        var = torch.stack([mvn.variance for mvn in predictions], -1)
        cov = torch.stack([torch.diag(item) for item in var])
        batch_mvn = gpytorch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        
        
        
        
        
        
        return batch_mvn

    def add_observations(self, x: torch.Tensor, y: torch.Tensor):
        assert x.ndim == 2 and x.shape[1] == self.in_dim
        assert y.ndim == 2 and y.shape[1] == self.out_dim
        if self.model.models[0].prediction_strategy is None:
            with torch.no_grad():
                self.predict(x, noiseless=True)
        rep_x = [x for _ in range(self.out_dim)]
        self.model = self.model.get_fantasy_model(rep_x, y.T)
        self.mll = SumMarginalLogLikelihood(self.likelihood, self.model)
        return

    def update_mean_constant(self):
        for i, m in enumerate(self.model.models):
            mean_i = self.train_targets[:, i].mean()
            m.mean_module.initialize(constant=mean_i)
            m.mean_module.constant.requires_grad = False
        return

    def __getstate__(self):
        state = self.__dict__.copy()
        model_state = self.model.state_dict()
        del state['model']
        del state['mll']
        state['model_state'] = model_state
        state['train_x'] = self.train_inputs
        state['train_y'] = self.train_targets
        return state

    def __setstate__(self, state):
        model_state = state['model_state'].copy()
        train_x = state['train_x'].clone()
        train_y = state['train_y'].clone()

        del state['model_state']
        del state['train_x']
        del state['train_y']

        self.__dict__.update(state)

        each_train_y = [train_y[:, i] for i in range(self.out_dim)]
        each_model = []
        each_mll = []
        for i in range(self.out_dim):
            if self.outputscale_bounds is None:
                os_bound = None
            elif len(self.outputscale_bounds) == 1:
                os_bound = self.outputscale_bounds[0]
            else:
                os_bound = self.outputscale_bounds[i]

            model_i = SingleTaskGP(train_x=train_x, train_y=each_train_y[i],
                                   outputscale_bounds=os_bound,
                                   lengthscale_bounds=self.lengthscale_bounds,
                                   likelihood=self.noise_var, use_ard=self.use_ard, prior_mean=self.prior_mean)
            mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
            each_model.append(model_i)
            each_mll.append(mll_i)
        self.model = IndependentModelList(*each_model)
        self.model.load_state_dict(model_state)
        self.mll = SumMarginalLogLikelihood(self.likelihood, self.model)
        return



class CascadeMOGP:
    def __init__(self, models: list[IndependentMOGP]):
        self.models: list[IndependentMOGP] = models
        assert models[-1].out_dim == 1
        return

    @property
    def n_stage(self):
        return len(self.models)

    @property
    def in_dims(self):
        return [item.in_dim for item in self.models]

    @property
    def out_dims(self):
        return [item.out_dim for item in self.models]

    @property
    def all_train_inputs(self) -> list[torch.Tensor]:
        return [item.train_inputs for item in self.models]

    @property
    def all_train_targets(self) -> list[torch.Tensor]:
        return [item.train_targets for item in self.models]

    def add_observations(self, i: int, x: torch.Tensor, y: torch.Tensor):
        self.models[i].add_observations(x, y)
        return

    def fit_hyper_param(self, i: int = None, max_retries: int = 10) -> None:
        if i is None:
            for m in self.models:
                m.update_mean_constant()
                fit_param(m.mll, max_retries=max_retries)
        else:
            self.models[i].update_mean_constant()
            fit_param(self.models[i].mll, max_retries=max_retries)
        return

    def predict(self, i: int, x: torch.Tensor, noiseless: bool = True, output_i: int = None):
        return self.models[i].predict(x, noiseless=noiseless, output_i=output_i)


class MultitaskGP(ExactGP):
    def __init__(self, train_x, train_y,
                 outputscale_bounds=None, lengthscale_bounds=None, use_ard: bool = False,
                 cor='icm', rank: int = 1):
        self.in_dim = train_x.shape[1]
        self.out_dim = train_y.shape[1]
        self.use_ard = use_ard
        likelihood = MultitaskGaussianLikelihood(num_tasks=self.out_dim, noise_constraint=Interval(1e-6, 1e-2),
                                                 has_task_noise=False)
        likelihood.noise.data = likelihood.raw_noise_constraint.inverse_transform(1e-4)
        likelihood.raw_noise.requires_grad = False
        
        

        super().__init__(train_x, train_y, likelihood)

        means = [ConstantMean() for _ in range(self.out_dim)]
        for i in range(self.out_dim):
            means[i].initialize(constant=train_y[:, i].mean())
            means[i].constant.requires_grad = False
        self.mean_module = MultitaskMean(
            means, num_tasks=self.out_dim
        )
        self.rank = rank
        if outputscale_bounds is None:
            outputscale_bounds = (0.01, 250.)
        if lengthscale_bounds is None:
            if use_ard:
                lengthscale_bounds = [(0.1, 1000.) for _ in range(train_y.shape[1])]
            else:
                lengthscale_bounds = (0.1, 1000.)
        if use_ard:
            kernels = []
            for i in range(self.in_dim):
                ls_bound = Interval(*lengthscale_bounds[i])
                kernels.append(RBFKernel(active_dims=(i,), lengthscale_constraint=ls_bound))
            self.rbf = ProductKernel(*kernels)
        else:
            if isinstance(lengthscale_bounds, list) and len(lengthscale_bounds) >= 1:
                lengthscale_bounds = lengthscale_bounds[0]
            self.rbf = RBFKernel(lengthscale_constraint=Interval(*lengthscale_bounds))
        if cor == 'icm':
            self.covar_module = MultitaskKernel(
                self.rbf, num_tasks=self.out_dim, rank=rank
            )
            
            self.covar_module.task_covar_module.raw_var_constraint = Interval(*outputscale_bounds)
        else:
            raise NotImplementedError()
        return

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def predict(self, x, noiseless: bool = False):
        pred = self(x)
        if not noiseless:
            pred = self.likelihood(pred)
        return pred

    def print_params(self, print_fixed_param: bool = False):
        if not print_fixed_param:
            means = [item.constant.data for item in self.mean_module.base_means]
            print(f'constant_means={torch.tensor(means)}')
            print(f'noise_var={self.likelihood.noise}')
            
        if not self.use_ard:
            ls = self.rbf.lengthscale
        else:
            ls = [float(k.lengthscale.detach()) for k in self.rbf.kernels]
        print(f'rbf:lengthscale={ls}')
        print(f'index:diag_var={self.covar_module.task_covar_module.var}')
        print(f'index:covar=\n{self.covar_module.task_covar_module.covar_matrix.detach().numpy()}')
        cor = cov_to_cor(self.covar_module.task_covar_module.covar_matrix.detach().numpy())
        print(f'index:cor_matrix=\n{cor}')
        return


class MultitaskLMCGP(ExactGP):
    def __init__(self, train_x, train_y,
                 outputscale_bounds=None, lengthscale_bounds=None, use_ard: bool = False,
                 cor='lmc', rank: int = 1):
        self.in_dim = train_x.shape[1]
        self.out_dim = train_y.shape[1]
        self.use_ard = use_ard
        likelihood = MultitaskGaussianLikelihood(num_tasks=self.out_dim, noise_constraint=Interval(1e-6, 1e-2),
                                                 has_task_noise=False)
        likelihood.noise.data = likelihood.raw_noise_constraint.inverse_transform(1e-4)
        likelihood.raw_noise.requires_grad = False
        
        

        super().__init__(train_x, train_y, likelihood)

        means = [ConstantMean() for _ in range(self.out_dim)]
        for i in range(self.out_dim):
            means[i].initialize(constant=train_y[:, i].mean())
            means[i].constant.requires_grad = False
        self.mean_module = MultitaskMean(
            means, num_tasks=self.out_dim
        )
        self.rank = rank
        if outputscale_bounds is None:
            outputscale_bounds = (0.01, 250.)
        if lengthscale_bounds is None:
            if use_ard:
                lengthscale_bounds = [(0.1, 1000.) for _ in range(train_y.shape[1])]
            else:
                lengthscale_bounds = (0.1, 1000.)
        if use_ard:
            rbf_kernels = []
            for i in range(self.in_dim):
                ls_bound = Interval(*lengthscale_bounds[i])
                rbf_kernels.append(RBFKernel(active_dims=(i,), lengthscale_constraint=ls_bound))
            self.rbf = ProductKernel(*rbf_kernels)
            matern_kernels = []
            for i in range(self.in_dim):
                ls_bound = Interval(*lengthscale_bounds[i])
                matern_kernels.append(MaternKernel(nu=1.5, active_dims=(i,), lengthscale_constraint=ls_bound))
            self.matern = ProductKernel(*matern_kernels)
        else:
            if isinstance(lengthscale_bounds, list) and len(lengthscale_bounds) >= 1:
                lengthscale_bounds = lengthscale_bounds[0]
            self.rbf = RBFKernel(lengthscale_constraint=Interval(*lengthscale_bounds))
            self.matern = MaternKernel(nu=1.5, lengthscale_constraint=Interval(*lengthscale_bounds))
        if cor == 'lmc':
            self.covar_module = LCMKernel(
                [self.rbf, self.matern], num_tasks=self.out_dim, rank=rank
            )
            
            for cov_module in self.covar_module.covar_module_list:
                cov_module.task_covar_module.raw_var_constraint = Interval(*outputscale_bounds)
        else:
            raise NotImplementedError()
        return

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def predict(self, x, noiseless: bool = False):
        pred = self(x)
        if not noiseless:
            pred = self.likelihood(pred)
        return pred

    def print_params(self, print_fixed_param: bool = False):
        if not print_fixed_param:
            means = [item.constant.data for item in self.mean_module.base_means]
            print(f'constant_means={torch.tensor(means)}')
            print(f'noise_var={self.likelihood.noise}')
            
        if not self.use_ard:
            rbf_ls = self.rbf.lengthscale
            matern_ls = self.matern.lengthscale
        else:
            rbf_ls = [float(k.lengthscale.detach()) for k in self.rbf.kernels]
            matern_ls = [float(k.lengthscale.detach()) for k in self.matern.kernels]
        print(f'rbf:lengthscale={rbf_ls}')
        print(f'matern:lengthscale={matern_ls}')
        for i, cov_module in enumerate(self.covar_module.covar_module_list):
            print(f'index:{i}:diag_var={cov_module.task_covar_module.var}')
            print(f'index:{i}:covar=\n{cov_module.task_covar_module.covar_matrix.detach().numpy()}')
            cor = cov_to_cor(cov_module.task_covar_module.covar_matrix.detach().numpy())
            print(f'index:{i}:cor_matrix=\n{cor}')
        return
