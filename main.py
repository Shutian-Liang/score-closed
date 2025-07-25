import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import set_weights, get_arguments
from src.score_solution import reverse_sde
from src.plotting import show_images, show_distribution, normalize, find_nearest_neighbors_manual

def main():
    args = get_arguments()
    weights = set_weights(args)
    sde = reverse_sde(args, weights)
    generated_path = f'./results/generated/{args.sde_version}/{args.start}/{args.dataset}'
    distance_path = f'./results/distance/'
    times = args.times

    # Create the directory if it doesn't exist
    if not os.path.exists(generated_path):
        os.makedirs(generated_path)
    
    if not os.path.exists(distance_path):
        os.makedirs(distance_path)

    # Save the generated images
    distance = np.array([])
    for time in range(times):
        print(f'sampling times : {time}')
        images = sde.p_sample_loop(return_all_time_steps=True)
        fig, axes, dis = show_images(images, args, sde)
        distance = np.append(distance, dis)
        fig.savefig(f'{generated_path}/sample_{time}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    # find the position of the top 3 biggest distance in the distance
    top_3_indices = np.argsort(distance)[-3:]
    sample_index = top_3_indices//args.batchsize
    pic_index = top_3_indices%args.batchsize
    
    # save as the json file
    with open(f'{distance_path}/{args.sde_version}_{args.start}_{args.dataset}.txt', 'w') as f:
        for i in range(len(top_3_indices)):
            f.write(f"Sample {sample_index[i]}, Image {pic_index[i]}, Distance: {distance[top_3_indices[i]]}\n")
    # print(f'Top 3 indices: {top_3_indices}')

    # Save the distance distribution
    fig, ax = show_distribution(distance)
    fig.savefig(f'{distance_path}/distribution_{args.sde_version}_{args.start}_{args.dataset}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()