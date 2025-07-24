import numpy as np 
from scipy.special import logsumexp
from tqdm import tqdm

class reverse_sde:
    """
    Reverse SDE class for handling the reverse process of the SDE.
    """
    def __init__(self, args, weights):
        self.args = args
        self.timesteps = args.timesteps
        self.betas = weights['betas']
        self.alphas_cumprod = weights['alphas_cumprod']
        self.sqrt_alphas_cumprod = weights['sqrt_alphas_cumprod']
        self.sqrt_one_minus_alphas_cumprod = weights['sqrt_one_minus_alphas_cumprod']
        self.posterior_sigmas = weights['posterior_sigmas']
        self.load_data() # load data
    
    def load_data(self):
        """
        Load the data for the reverse SDE.
        """
        # Implement data loading logic here
        data = np.load('../data/cifar10_1k.npz')
        self.images = data['images']
    
    def q_sample(self, x, t):
        """
        Sample from the reverse SDE at time t.
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Sample noise
        noise = np.random.normal(size=x.shape)
        
        # Compute the sample
        xt =  sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise

        return xt
    
    def compute_weights(self, xt, t):
        """
        Compute the probability density function at time t.
        """
        # assert xt.shape == self.images.shape, "Input shape must match the images shape"
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        # xt = self.q_sample(x, t)
        xt = xt[:, np.newaxis]
        
        y = self.images * sqrt_alpha_cumprod
        sigma = sqrt_one_minus_alpha_cumprod
        logp = -(xt - y)**2 / (2 * sigma**2) - np.log(np.sqrt(2 * np.pi * sigma**2))
        logp = np.sum(logp, axis=(2, 3, 4))  # Sum over all dimensions shaped [b,n]

        log_sum_of_probs = logsumexp(logp, axis=1, keepdims=True)  # Sum over all dimensions except batch
        log_weights = logp - log_sum_of_probs  # Normalize log probabilities
        weights = np.exp(log_weights)
        # print(weights.shape, logp.shape)
        assert weights.shape == logp.shape, "Weights shape must match log probabilities shape"
        return weights, xt, y, sigma
    
    def score_function(self, xt, t):
        """
        Compute the score function at time t.
        """
        weights, xt, y, sigma = self.compute_weights(xt, t)
        if t % 100 == 0: # 每100步打印一次
            print(f"t={t}, max_weight={np.max(weights, axis=1).mean()}")
        grad = -(xt - y) / (sigma**2)
        weights = weights[:,:, np.newaxis, np.newaxis, np.newaxis]  # Reshape weights to match grad shape
        scores = np.sum(grad * weights, axis=1)
        xt = np.squeeze(xt, axis=1)  # Remove the extra dimension

        assert scores.shape == xt.shape, "Scores shape must match input shape"
        return scores,xt

    def p_sample(self, xt, t):
        """
        Sample from the reverse SDE at time t.
        """
        scores, xt = self.score_function(xt, t)
        alpha_cumprod = self.alphas_cumprod[t]  
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        # posterior_sigma = self.posterior_sigmas[t]
        posterior_sigma = self.betas[t]
        
        # Sample noise
        noise = np.random.normal(size=xt.shape)
        img = 1./sqrt_alpha_cumprod * ( xt + (1 - alpha_cumprod) * scores) + posterior_sigma * noise
        return img

    def p_sample_loop(self, shape, return_all_time_steps=True):
        img = np.random.randn(*shape).astype(np.float32)
        imgs = [img]
        for t in tqdm(reversed(range(0, self.timesteps)), desc = 'sampling loop time step', total = self.timesteps):
            img = self.p_sample(img, t)
            imgs.append(img)
            
        return imgs[-1] if not return_all_time_steps else imgs
    