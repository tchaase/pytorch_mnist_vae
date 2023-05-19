# This section contains utility functions related to saving plots and images. They should not clutter up the training part as they are not relevant to training. 

import imageio #This is chosen because it allows saving the pictures as .gif 
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

to_pil_image = transforms.ToPILImage() #conversion of a tensor to an image, this will later be used to generate .gif images!

#The following function converts the PIL images to .gif files. The accepted data are numpy arrays!
def image_to_vid(images):     
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('../outputs/generated_images.gif', imgs)

#This function is equal to the function above, but for outputs of the VAE
def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"../outputs/output{epoch}.jpg")  #The save image function comes from torchvision.utilis!

#Finally, here the training and validation losses are saved into a plot! This is done via matplotlib. 
def save_loss_plot(train_loss, valid_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.jpg')
    plt.show()