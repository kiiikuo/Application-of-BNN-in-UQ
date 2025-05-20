# visualize.py

import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(mean_pred, var_pred, ground_truth, idx=0):
    
    mean_np = mean_pred[idx].cpu().numpy()
    var_np = var_pred[idx].cpu().numpy()
    gt_np = ground_truth[idx].cpu().numpy()
    error_np = np.abs(mean_np - gt_np)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Predicted Mean
    im0 = axs[0].imshow(mean_np, cmap='coolwarm', origin='lower')
    axs[0].set_title("Predicted Mean")
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    fig.colorbar(im0, ax=axs[0], orientation='vertical')

    # Predictive Variance
    im1 = axs[1].imshow(var_np, cmap='viridis', origin='lower')
    axs[1].set_title("Predictive Variance")
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    fig.colorbar(im1, ax=axs[1], orientation='vertical')

    # Absolute Error
    im2 = axs[2].imshow(error_np, cmap='hot', origin='lower')
    axs[2].set_title("Absolute Error")
    axs[2].set_xlabel('Longitude')
    axs[2].set_ylabel('Latitude')
    fig.colorbar(im2, ax=axs[2], orientation='vertical')

    plt.tight_layout()
    plt.show()
