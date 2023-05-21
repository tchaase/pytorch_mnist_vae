# This script defines funcitons that are criticical to the process of training the VAE. 

from tqdm import tqdm
import torch 

# Firstly, the loss function is defined. The KL divergence is computed manually. There are three inputs: Firstly the reconstruction loss. 
# Then there are the mean and the variance, which are related to the VAE's latent space.
# The loss is defined as the sum of the KL divergence and the reconstruction loss here, thus the sum is returned as the final loss.  
def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Next, the training function is defined. This takes the criterion refers to the loss function, which is the criterion that needs to be minimized. 
def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1  #There is a counter that is initialized at 0, which goes up for every batch?
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()  # Using the loss, backpropagation occurs. Thus, all the tensors that will be connected to this, will be involved in this computation. 
        running_loss += loss.item() #This here defined for every step along the way, how high is the loss. 
        optimizer.step()  #That with a gradient will be updated in one step according to the documentation. 
    train_loss = running_loss / counter 
    return train_loss #The function returns only the training loss. 


# Lastly, we need to validate that the training worked. Two things differentiate this from the training: Firstly, there is no backpropagation step here! The evaluation
# does not impact the training. Then, images are saved. This is saved then according to the functions outlined in the utilis section. 
def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images