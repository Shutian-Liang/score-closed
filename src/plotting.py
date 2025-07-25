import torch
import numpy as np
import matplotlib.pyplot as plt

def normalize(imgs):
    """
    Normalize the images to the range [0, 1]
    """
    imgs = imgs.cpu().detach().numpy()
    imgs =  (imgs+1)*0.5
    imgs = np.clip(imgs, 0, 1)
    return imgs

def find_nearest_neighbors_manual(query_images, database_images):
    """
    L2  distance for computing
    """
    q = query_images.unsqueeze(1)
    d = database_images.unsqueeze(0)
    
    dist_sq = torch.sum((q - d) ** 2, dim=(2, 3, 4))
    
    dist_matrix = torch.sqrt(dist_sq) # shape (n, m)
    nearest_distances, nearest_indices = torch.min(dist_matrix, dim=1)
    
    return nearest_distances, nearest_indices

def show_images(images, args, sde):
    """
    Show images in a grid,行列转置
    """
    fig, axes = plt.subplots(args.batchsize, 13, figsize=(10, 5)) 
    dis, indices = find_nearest_neighbors_manual(images[-1].to(sde.device), sde.images)
    dis = dis.cpu().detach().numpy()
    # print(dis)
    neighbor = normalize(sde.images[indices])
    # sampling trajetory
    for i in range(11):
        index = max(0, i*100-1)
        imgs = normalize(images[index])
        for j in range(args.batchsize):
            ax = axes[j, i]  # 行列转置
            ax.imshow(imgs[j].transpose(1, 2, 0))
            ax.axis('off')
            if j == 0:
                ax.set_title(f"t = {index}", fontsize=10)
    
    # nearest neighbor
    for j in range(args.batchsize):
        ax = axes[j, 11]
        ax.imshow(neighbor[j].transpose(1, 2, 0))
        ax.axis('off')
        if j == 0:
            ax.set_title("Nearest", fontsize=10)
    
    if args.start == 'sub_noise':
        # 显示原始图像
        original_imgs = normalize(sde.sample)
        for j in range(args.batchsize):
            ax = axes[j, 12]
            ax.imshow(original_imgs[j].transpose(1, 2, 0))
            ax.axis('off')
            if j == 0:
                ax.set_title("Image", fontsize=10)
    else:
        for j in range(args.batchsize):
            ax = axes[j, 12]
            # ax.imshow(original_imgs[j].transpose(1, 2, 0))
            ax.axis('off')
            
    plt.suptitle(f'start {args.start}')
    plt.subplots_adjust(wspace=0.2)
    return fig, axes, dis

def show_distribution(dis):
    """
    Show the distribution of nearest neighbor distances
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.hist(dis, bins=50, density=True, alpha=0.6, color='g')
    plt.title('Distribution of Nearest Neighbor Distances')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    # plt.savefig(f'distribution_{args.sde_version}_{args.start}_{args.dataset}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    return fig, ax