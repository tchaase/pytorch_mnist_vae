# --- Preparation
import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot
matplotlib.style.use('ggplot')
# Firstly, the device is set to the GPU instead of the CPU for me, as I have cuda available. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model - here a convolutional VAE is used which will be computed on my GPU. What does this mean? It uses a convolution to analyze the images. 
model = model.ConvVAE().to(device)
# set the learning parameters
lr = 0.001  # I have no idea if this is large. 
epochs = 100
batch_size = 64
# Trained 100 rounds, 64 samples each episode.  
optimizer = optim.Adam(model.parameters(), lr=lr)  # An ADAM optimizer is used. This means that the optimization occurs in one step, although I still need to read more on what this does exactly. 
criterion = nn.BCELoss(reduction='sum')  #BCE stands for Binary Cross Entropy, thus cross entropy is used as the criterion. 
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []  #An empty list is created. 

# --- Data Transformation: 
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
# The original data-set available from MNIST have a size of 28x28 pixels. Above they are transformed. 

# --- training set and train data loader
trainset = torchvision.datasets.MNIST(
    root='../input', train=True, download=True, transform=transform
)
# The data-set is loaded. It comes from the source of torchvision.data.set and is downloaded. 
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
# Here is the important set of actually loading the data. The batch sizes are also defined, thus it is defined how data will be sampled for later training?

# --- validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root='../input', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)
# Again empty lists are initialized for the training loss and the validation loss to allow for later plotting. 
train_loss = []
valid_loss = []
# Here the training function is defined. For every epoch in the range of epochs the following function will be computed. This means that for the first episode
# I will draw from the training set, compute the loss, backpropagate and rinse and repeat. 
# The bread and butter of this are the train and validate functions. they require the input of the model, what data will be used to train, what data will be used to test. 
# What is the device (for me GPU), then how will it be optimized (ADAM), in regards to the previously defined criterion of binary cross entropy. 
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, testloader, testset, device, criterion
    )
    # Next the values that result from this will be stored. 
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format, which can then be saved!
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}") #Up to 4 decimals will be printed, float format. 

# save the reconstructions as a .gif file
image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
print('TRAINING COMPLETE')