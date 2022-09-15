import torch
from torch.distributions import Normal

from models import CascadeMOGP, SingleTaskGP


def _ei(mean: torch.Tensor, std: torch.Tensor, current_best: torch.Tensor, maximize: bool = True) -> torch.Tensor:
    z = (mean - current_best) / std
    if not maximize:
        z *= -1
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    z_cdf = normal.cdf(z)
    z_pdf = torch.exp(normal.log_prob(z))  
    ei = std * (z_pdf + z * z_cdf)
    return ei


class CascadeEI:

    def __init__(self, model: CascadeMOGP, current_best, base_samples: list[torch.Tensor], maximize: bool = True):
     

        if not torch.is_tensor(current_best):
            current_best = torch.tensor(current_best, dtype=torch.float)
        self.current_best = current_best
        self.model = model
        self.maximize = maximize
        assert len(base_samples) == model.n_stage
        n_sample = base_samples[0].shape[0]
        assert all([item.shape[0] == n_sample for item in base_samples])
        self.base_samples = base_samples
        self.n_sample = n_sample
        return

    def evaluate(self, obs_stage: int, x: torch.Tensor) -> torch.Tensor:
    

        if x.ndim == 1:
            x = x.reshape(1, -1)

        nx = x.shape[0] 

        if obs_stage == self.model.n_stage - 1:
            pred = self.model.predict(obs_stage, x, noiseless=True)
            mean = pred.mean.flatten()
            std = pred.stddev.flatten()
            ei = _ei(mean, std, current_best=self.current_best, maximize=self.maximize)
            return ei
        else:

            i = obs_stage

            bs_i = torch.tile(self.base_samples[i], (nx, 1, 1)).transpose(0, 1)  
            input_len = self.model.in_dims[i] 
            xi = x[:, :input_len]
            pred = self.model.predict(obs_stage, xi, noiseless=False) 
            sample_y = pred.rsample(bs_i.shape, base_samples=bs_i) 

            prev_out_dim = sample_y.shape[-1]  
            sum_len = input_len  
            i += 1

            
            while i < self.model.n_stage - 1:
                input_len = self.model.in_dims[i] - prev_out_dim  

                xi = x[:, sum_len:sum_len + input_len]  
                tile_x = torch.tile(xi, (self.n_sample, 1, 1))  
                input_x = torch.cat([sample_y, tile_x], dim=2)  

                
                sample_y = torch.empty((self.n_sample, nx, self.model.out_dims[i]))  
                for j in range(nx):
                    pred = self.model.predict(i, input_x[:, j, :], noiseless=False)
                    sample_y[:, j, :] = pred.rsample(self.base_samples[i].shape,
                                                     base_samples=self.base_samples[i])  
                prev_out_dim = sample_y.shape[-1]
                sum_len += input_len
                i += 1

            
            input_len = self.model.in_dims[i] - prev_out_dim  
            xi = x[:, sum_len:sum_len + input_len]  
            tile_x = torch.tile(xi, (self.n_sample, 1, 1))  
            input_x = torch.cat([sample_y, tile_x], dim=2)  

            cascade_ei = torch.zeros(nx, )  
            
            for j in range(nx):
                pred = self.model.predict(i, input_x[:, j, :], noiseless=False)  
                mean = pred.mean.flatten()  
                std = pred.stddev.flatten()  
                ei_each_sample = _ei(mean, std, current_best=self.current_best, maximize=self.maximize)
                cascade_ei[j] = torch.mean(ei_each_sample)
            return cascade_ei


class CascadeUCB:

    def __init__(self, model: CascadeMOGP, root_beta, lipschitz=1., lcb: bool = False):
       
        self.model = model
        self.lcb = lcb
        self.n_stage = model.n_stage

        if not torch.is_tensor(root_beta):
            root_beta = torch.tensor(root_beta, dtype=torch.float64).flatten()
        if len(root_beta) == 1:  
            
            root_beta = torch.full((self.n_stage,), float(root_beta), dtype=torch.float64)
        elif self.n_stage != len(root_beta):
            raise ValueError('len(root_beta) should be n_stage or 1.')
        self.root_beta = root_beta

        if not torch.is_tensor(lipschitz):
            lipschitz = torch.tensor(lipschitz, dtype=torch.float64).flatten()
        if len(lipschitz) == 1:  
            
            lipschitz = torch.full((self.n_stage,), float(lipschitz), dtype=torch.float64)
        elif self.n_stage != len(lipschitz):
            raise ValueError('len(lipschitz) should be n_stage or 1.')
        self.lipschitz = lipschitz
        return

    def evaluate(self, obs_stage: int, x: torch.Tensor) -> torch.Tensor:
   
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        nx = x.shape[0]  
        prev_out_dim = 0  
        sum_len = 0

        
        prev_mu_tilde = None
        
        prev_sigma_tilde_sum = torch.zeros((nx,), dtype=torch.float64)

        for i in range(obs_stage, self.n_stage):
            input_len = self.model.in_dims[i] - prev_out_dim  
            xi = x[:, sum_len:sum_len + input_len]  
            if i > obs_stage:
                xi = torch.cat([prev_mu_tilde, xi], dim=1)  
            pred = self.model.predict(i, xi, noiseless=True)  

            mu_tilde = pred.mean  
            sigma_tilde = pred.stddev + self.lipschitz[i] * prev_sigma_tilde_sum[:, None]  
            if i == self.n_stage - 1:
                ci_length = self.root_beta[i] * sigma_tilde  
                if self.lcb:
                    return (mu_tilde - ci_length).flatten()
                else:
                    return (mu_tilde + ci_length).flatten()
            prev_mu_tilde = mu_tilde
            prev_sigma_tilde_sum = sigma_tilde.sum(dim=1)
            prev_out_dim = mu_tilde.shape[1]
            sum_len += input_len

        
        raise RuntimeError('Function values should have been returned before reaching this line.')


class CascadeUncertainty:
   

    def __init__(self, model: CascadeMOGP, lipschitz=1.):
        self.model = model
        self.n_stage = model.n_stage

        if isinstance(lipschitz, (float, int)):
            lipschitz = torch.full((self.n_stage,), lipschitz, dtype=torch.float64)
        if not torch.is_tensor(lipschitz):
            lipschitz = torch.tensor(lipschitz, dtype=torch.float64)
        self.lipschitz = lipschitz
        return

    def evaluate(self, obs_stage: int, x: torch.Tensor):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        nx = x.shape[0]
        prev_out_dim = 0
        past_index = 0

        add_term = torch.zeros((nx,), dtype=torch.float64)  
        prev_mean = None
        for i in range(obs_stage, self.n_stage):
            input_len = self.model.in_dims[i] - prev_out_dim
            xi = x[:, past_index:past_index + input_len]  
            if i > obs_stage:
                xi = torch.cat([prev_mean, xi], dim=1)
            pred = self.model.predict(i, xi, noiseless=True)  
            mean = pred.mean  
            std = pred.stddev  

            ci_length = std + add_term[:, None]  
            if i == self.n_stage - 1:
                return torch.square(ci_length).flatten()
            prev_mean = mean
            add_term = self.lipschitz[i] * ci_length.sum(dim=1)
            prev_out_dim = mean.shape[1]
            past_index += input_len

        
        raise RuntimeError('Function values should have been returned before reaching this line.')


class CBO:

    def __init__(self, model: CascadeMOGP, current_best, k1=1.0, k2=1.0, maximize: bool = True):
       
        if not torch.is_tensor(current_best):
            current_best = torch.tensor(current_best, dtype=torch.float)
        if not torch.is_tensor(k1):
            k1 = torch.tensor(k1, dtype=torch.float)
        if not torch.is_tensor(k2):
            k2 = torch.tensor(k2, dtype=torch.float)
        self.current_best = current_best
        self.k1 = k1
        self.k2 = k2
        self.model = model
        self.maximize = maximize
        return

    def evaluate(self, obs_stage: int, x: torch.Tensor, desired_prev_y: torch.Tensor = None):
        
        if obs_stage != self.model.n_stage - 1:
            assert desired_prev_y is not None
            desired_prev_y = torch.flatten(desired_prev_y)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        nx = x.shape[0]
        out_dim = self.model.out_dims[obs_stage]
        

        if obs_stage == self.model.n_stage - 1:
            pred = self.model.predict(obs_stage, x, noiseless=True)
            mean = pred.mean.flatten()
            std = pred.stddev.flatten()
            ei = _ei(mean, std, current_best=self.current_best, maximize=self.maximize)
            return ei
        else:
            pred = self.model.predict(obs_stage, x, noiseless=False)  
            mean = pred.mean.reshape(nx, out_dim)  
            scale_tril = pred.scale_tril.reshape(nx, out_dim, out_dim)
            cbo_value = self._cbo_value(mean, scale_tril, desired_prev_y).reshape(-1)
            return cbo_value

    def _cbo_value(self, mean: torch.Tensor, scale_tril: torch.Tensor, desired_y: torch.Tensor):
     
        inv_tril = torch.triangular_solve(torch.eye(mean.shape[1]), scale_tril, upper=False)[0]  
        base_diff = (desired_y - mean)[..., None]  

        diff_k1 = torch.bmm(inv_tril, base_diff).squeeze(-1)
        diff_k2 = torch.bmm(scale_tril, base_diff).squeeze(-1)
        sq_norm_k1 = torch.square(torch.linalg.norm(diff_k1, dim=1))
        sq_norm_k2 = torch.square(torch.linalg.norm(diff_k2, dim=1))

        cbo = (self.k1 * sq_norm_k1 + self.k2 * sq_norm_k2)
        return cbo


class EI:

    def __init__(self, model: SingleTaskGP, current_best, maximize: bool = True):
      
        self.model = model
        self.maximize = maximize
        if not torch.is_tensor(current_best):
            current_best = torch.tensor(current_best, dtype=torch.float)
        self.current_best = current_best
        return

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
       
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pred = self.model.predict(x, noiseless=True)
        mean = pred.mean
        std = pred.stddev
        ei = _ei(mean, std, current_best=self.current_best, maximize=self.maximize)
        return ei


class UCB:

    def __init__(self, model: SingleTaskGP, root_beta=3, lcb: bool = False):
       
        self.model = model
        self.lcb = lcb
        if not torch.is_tensor(root_beta):
            root_beta = torch.tensor(root_beta, dtype=torch.float64)
        self.root_beta = root_beta.reshape((1,))
        return

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
      
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pred = self.model.predict(x, noiseless=True)
        mean = pred.mean
        std = pred.stddev
        if self.lcb:
            return mean - self.root_beta * std
        else:
            return mean + self.root_beta * std



class EIFunctionNetwork:
   
    def __init__(self, model: CascadeMOGP, current_best, base_samples: list[torch.Tensor], maximize: bool = True):
        if not torch.is_tensor(current_best):
            current_best = torch.tensor(current_best, dtype=torch.float)
        self.current_best = current_best
        self.model = model
        self.maximize = maximize
        assert len(base_samples) == model.n_stage
        n_sample = base_samples[0].shape[0]
        assert all([item.shape[0] == n_sample for item in base_samples])
        self.base_samples = base_samples
        self.n_sample = n_sample
        return

    def evaluate(self, obs_stage: int, x: torch.Tensor):
   
        if x.ndim == 1:
            x = x.reshape(1, -1)
        nx = x.shape[0]

        i = obs_stage
        bs_i = torch.tile(self.base_samples[i], (nx, 1, 1)).transpose(0, 1)

        input_len = self.model.in_dims[i]
        past_index = 0
        xi = x[:, :input_len]
        pred = self.model.predict(obs_stage, xi, noiseless=(i == self.model.n_stage - 1))
        sample_y = pred.rsample(bs_i.shape, base_samples=bs_i)
        prev_out_dim = sample_y.shape[-1]
        i += 1

        while i < self.model.n_stage:
            past_index += input_len
            input_len = self.model.in_dims[i] - prev_out_dim

            xi = x[:, past_index:past_index + input_len]
            tile_x = torch.tile(xi, (self.n_sample, 1, 1))
            sample_x = torch.cat([sample_y, tile_x], dim=2)

            sample_y = torch.empty((self.n_sample, nx, self.model.out_dims[i]))
            for j in range(nx):
                pred = self.model.predict(i, sample_x[:, j, :], noiseless=(i == self.model.n_stage - 1))
                sample_y[:, j, :] = pred.rsample(self.base_samples[i].shape, self.base_samples[i])
            prev_out_dim = sample_y.shape[-1]
            i += 1

        sample_y = sample_y.reshape((self.n_sample, nx))
        diff = sample_y - self.current_best
        if not self.maximize:
            diff *= -1
        improvement = torch.clamp(diff, min=0)
        mean_ei = torch.mean(improvement, dim=0)
        return mean_ei
