import torch
from tqdm import tqdm
import numpy as np # Keep numpy for data loading

class reverse_sde:
    """
    Reverse SDE class implemented in PyTorch.
    """
    def __init__(self, args, weights):
        self.args = args
        self.timesteps = args.timesteps
        
        self.device = 'cuda:' + str(self.args.device) if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.betas = torch.from_numpy(weights['betas']).float().to(self.device)
        self.alphas_cumprod = torch.from_numpy(weights['alphas_cumprod']).float().to(self.device)
        self.sqrt_alphas_cumprod = torch.from_numpy(weights['sqrt_alphas_cumprod']).float().to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(weights['sqrt_one_minus_alphas_cumprod']).float().to(self.device)
        self.posterior_sigmas = torch.from_numpy(weights['posterior_sigmas']).float().to(self.device)
        
        self.load_data()
        self.set_sample()

    def load_data(self):
        """
        Load data and convert it to PyTorch tensors.
        """
        data = np.load('../data/cifar10_2k.npz')
        self.images = torch.from_numpy(data['images']).float().to(self.device)
    
    def set_sample(self):
        """
        whether use pure noise or data as the initial sample
        """
        if self.args.start == 'noise':
            self.img = torch.randn(self.args.batchsize, 3, 32, 32, device=self.device)
        elif self.args.start == 'sub_noise':
            img_number = np.random.randint(0, self.images.shape[0], self.args.batchsize)
            self.sample = self.images[img_number].clone().to(self.device)
            noise = torch.randn_like(self.sample)
            self.img = self.sqrt_alphas_cumprod[-1]*self.sample + self.sqrt_one_minus_alphas_cumprod[-1]*noise
            self.img = self.sample + 0.5*noise  # Use the sample as the base and add noise

    def q_sample(self, x, t):
        """
        Sample from the forward process at time t.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(self.device)
            
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        noise = torch.randn_like(x)
        
        xt = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        return xt

    def compute_weights(self, xt, t):
        """
        Compute the probability density function at time t.
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        xt = xt[:, None] # Equivalent to np.newaxis for broadcasting
        
        y = self.images * sqrt_alpha_cumprod
        sigma = sqrt_one_minus_alpha_cumprod
        
        logp = -(xt - y)**2 / (2 * sigma**2) - torch.log(torch.sqrt(2 * torch.pi * sigma**2))
        logp = torch.sum(logp, dim=(2, 3, 4)) # Shape [b, n]

        log_sum_of_probs = torch.logsumexp(logp/self.args.temperature, dim=1, keepdim=True)
        log_weights = (logp/self.args.temperature) - log_sum_of_probs
        weights = torch.exp(log_weights)
        
        assert weights.shape == logp.shape, "Weights shape must match log probabilities shape"
        return weights, xt, y, sigma

    def score_function(self, xt, t):
        """
        Compute the score function at time t.
        """
        weights, xt, y, sigma = self.compute_weights(xt, t)
        if (t%1 == 0) or (t == self.timesteps - 1):
            print(f"t={t}, {weights.max(dim=1).indices}")
            
        grad = -(xt - y) / (sigma**2)
        
        weights = weights[:, :, None, None, None] # Reshape weights for broadcasting
        scores = torch.sum(grad * weights, dim=1)
        xt = torch.squeeze(xt, dim=1) # Remove the extra dimension

        assert scores.shape == xt.shape, "Scores shape must match input shape"
        return scores, xt

    def p_sample(self, xt, t):
        """
        Sample from the reverse process at time t.
        """
        scores, xt = self.score_function(xt, t)
        alpha_cumprod = self.alphas_cumprod[t]  
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        # posterior_sigma = self.posterior_sigmas[t]
        posterior_sigma = torch.sqrt(self.betas[t])
        
        noise = torch.randn_like(xt)
        
        img = 1./sqrt_alpha_cumprod * (xt + (1 - alpha_cumprod) * scores) + posterior_sigma * noise
        return img

    def p_sample_loop(self, return_all_time_steps=True):
        """
        Execute the full reverse sampling loop.
        """
        img = self.img
        imgs = [img]
        
        for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            with torch.no_grad():
                img = self.p_sample(img, t)
            
            if return_all_time_steps:
                imgs.append(img.cpu())
        
        final_img = imgs[-1].cpu() if return_all_time_steps else img.cpu()
        return final_img if not return_all_time_steps else imgs