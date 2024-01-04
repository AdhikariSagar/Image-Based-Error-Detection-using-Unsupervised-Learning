import numpy as np
import os
import torch
import matplotlib.pyplot as plt
def plot_generated_images(data_loader, model, device, 
                      unnormalizer=None,
                      figsize=(12, 5), n_images=8, modeltype='CAE'):

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    plt.subplots_adjust(wspace=None, hspace=None)
    

    for batch_idx, (features, _) in enumerate(data_loader):

        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]

        with torch.no_grad():
            if modeltype == 'CAE':
                decoded_images = model(features)[:n_images]
            elif modeltype == 'VAE':
                encoded, z_mean, z_log_var, decoded_images = model(features)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
                
                ax[i].axis('off')
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='gray')
                ax[i].axis('off')
            plt.tight_layout(pad=0.)
            
    