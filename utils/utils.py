import os
import numpy as np
import argparse

def get_arguments():
    """
    get the arguments from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default='0', help='Device to use for training')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size for training')
    
    # sde version
    # vp for ddpm and ve for smld
    parser.add_argument('--sde_version', type=str, default='vp', choices=['vp','ve'], help='Type of SDE to use')
    
    # starting point
    # sub_noise for signals, noise for gaussian noise
    parser.add_argument('--start', type=float, default='sub_noise', choices=['sub_noise','noise'], help='Starting point for the SDE')
    
    # sampling parameters
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps for the SDE')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    
    args = parser.parse_args()
    # get arguments
    return args

def vp_weights(timesteps):
    """
    Get the weights for the VP SDE
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    posterior_sigmas = np.sqrt(betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_sigmas': posterior_sigmas
    }

def ve_weights(timesteps):
    """
    Get the weights for the VE SDE, 
    similar to smld in yang song 2019
    """
    scale = 1000 / timesteps
    alphas = np.ones(timesteps).dtype(np.float32)
    alphas_cumprod = np.cumprod(alphas)
    sigma_min = scale * 0.01
    sigma_max = scale * 10.0
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    posteriot_sigmas = np.geomspace(sigma_min, sigma_max, timesteps).astype(np.float32)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - sigmas)
    
    return {
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_sigmas': posteriot_sigmas
    }

def set_weights(args):
    """
    Set the weights for the SDE based on the arguments
    """
    timesteps, sde_version = args.timesteps, args.sde_version
    if sde_version == 'vp':
        return vp_weights(timesteps)
    elif sde_version == 've':
        return ve_weights(timesteps)

class Namespace:  
    def __init__(self, **kwargs):  
        """
        use this class to create a namespace in jupyter notebook for debugging
        """
        self.__dict__.update(kwargs)    

def checkandcreate(path):
    """
    check if the path exists, if not create it
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory {path}')
    else:
        print(f'Directory {path} already exists')