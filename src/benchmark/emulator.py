import gpytorch
from gpytorch.constraints import Interval


class GPRegression(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood,
                 outputscale_bounds=None, lengthscale_bounds=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=train_y.mean())
        self.mean_module.constant.requires_grad = False

        if outputscale_bounds is None:
            outputscale_bounds = (0.01, 250.)
        if lengthscale_bounds is None:
            lengthscale_bounds = [(0.1, 1000.) for _ in range(train_x.shape[1])]

        in_dim = train_x.shape[1]
        kernels = []
        for i in range(in_dim):
            ls_bound = Interval(*lengthscale_bounds[i])
            kernels.append(gpytorch.kernels.RBFKernel(active_dims=(i,), lengthscale_constraint=ls_bound))
        self.rbf = gpytorch.kernels.ProductKernel(*kernels)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.rbf, outputscale_constraint=Interval(*outputscale_bounds))
        return

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def print_params(self):
        print(f'constant_mean={self.mean_module.constant}')
        print(f'noise_var={self.likelihood.noise}')
        print(f'output_scale={self.covar_module.outputscale}')
        ls = [float(k.lengthscale.detach()) for k in self.rbf.kernels]
        print(f'rbf:lengthscale={ls}')
        return

    def predict(self, x):
        return self(x)
