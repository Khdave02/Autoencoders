import matplotlib.pyplot as plt  # for plotting images
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm  # for showing progress bars during training and testing
import torch
import loader  # module for loading dataset and dataloader
seed = 1  # seed is used to ensure that we get the same output every time
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)
train_loader = loader.trainLoader(600)
test_loader = loader.testLoader(600)
lr =1e-3
num_epochs=9
class K_SparseAutoencoder(nn.Module):
    '''
            CNN model for the autoencoder
    '''
    def __init__(self):
        super().__init__()

        # encoding layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2,padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2,padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 5,stride=2,padding=1),
            nn.ReLU(),
        )
        self.enc4 = nn.Sequential(
            nn.Linear(576,256),
            nn.ReLU(),
        )

        # decoding layers
        self.dec1 = nn.Sequential(
            nn.Linear(256,576),
            nn.ReLU(),
        )
        self.dec2= nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5,stride=2,padding=1),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2,padding=1,output_padding=1),
            nn.ReLU(),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 3, stride=2,padding=1,output_padding=1),
            nn.ReLU()
        )

    # forward prop function
    def forward(self, x, k):
        '''
        returns activations of hidden layers as a list along with output.
        Takes one parameter:
        x: The input to the neural network
        k: k_sparse parameter to find k highest activation in bottleneck layer
        '''
        ret_val=[]  

        # encoder
        x = self.enc1(x)
        ret_val.append(x)
        x = self.enc2(x)
        ret_val.append(x)
        x = self.enc3(x)
        ret_val.append(x)
        x = x.view(-1,64*3*3)
        x = self.enc4(x)

        #bottle neck layer 
        topk,indices = torch.topk(x, k)
        x = torch.zeros(x.size())
        x = x.scatter_(1,indices,topk)
        ret_val.append(x)

        # decoder
        x = self.dec1(x)
        ret_val.append(x)
        x = x.view(-1,64,3,3)
        x = self.dec2(x)
        ret_val.append(x)
        x = self.dec3(x)
        ret_val.append(x)
        x = self.dec4(x)
        ret_val.append(x)
        return x,ret_val

def sparse_loss(model_children):
    '''
    calculates the sparse loss
    Calculates the L1 loss of the hidden layers passed as a list as model_children.
    '''
    loss = 0
    lamda=1e-4
    for i in model_children:
        loss += torch.mean(torch.abs(i))
    return lamda*loss

def training(model,trainloader,train_output,epoch,optimizer,criterion,k_sparse):
    '''
        trains the Neural Network.
        Parameters-
        model: CNN AutoEncoder model
        dataloader: DataLoader object that iterates through the train set
        opt: optimizer object to be used for training
        criterion: MSE loss function to be used for training
        train_outputs: list containing per epoch results(input & output) obtained after training
        epoch: current epoch
        Returns: train loss per epoch
    '''
    print("Training")
    loss = 0.0                  # Looping for every epoch
    for img,_ in tqdm(trainloader):  # Looping for every batch
            optimizer.zero_grad()  # Model training starts here
            
            outputs,model_children=model(img,k_sparse)
            train_loss = criterion(outputs, img) + sparse_loss(model_children)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
    loss = loss / len(trainloader)
    train_output.append((epoch, img, outputs))
    return loss


def testing(model,testloader,test_output,epoch,criterion,k_sparse):

    '''
            tests the trained model
            Parameters-
            model: CNN AutoEncoder model
            dataloader: DataLoader object that iterates through the test set
            criterion: MSE loss function to be used for testing
            test_outputs: list containing per epoch results(input & output) obtained after testing
            epoch: current epoch
            Returns: test loss per epoch
    '''
    print("Testing")
    with torch.no_grad():
            loss = 0.0
            for img,_ in tqdm(testloader):
                
                outputs,model_children=model(img,k_sparse)
                loss = criterion(outputs, img) 
                loss += loss.item()
            loss = loss / len(testloader)
            test_output.append((epoch, img, outputs))
            return loss


def cost_graph(loss_list,title):
    '''
    plots the loss graph of the test and train losses
    loss_list: loss per epoch
    title: title for the graph
    '''
    plt.plot(loss_list)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title(title)
    plt.show()

def view_images(output,num_epochs,k):
    '''
        displays the result images
        param output: list containing per epoch results(input & output) obtained after training/testing
    '''
    for k in range(0,num_epochs):
        plt.figure(figsize=(18, 5))
        plt.gray()
        ori_imgs = output[k][1]
        recon = output[k][2]
        n= 9
        for i in range(n):
            plt.subplot(2, n, i+1)
            plt.title("Original")
            plt.imshow(ori_imgs[i].to(torch.device('cpu')).detach().numpy().reshape(28,28),cmap = "gray")
            plt.subplot(2, n, i+1+n)
            plt.title("Sparse {}".format(k))
            plt.imshow(recon[i].to(torch.device('cpu')).detach().numpy().reshape(28,28),cmap = "gray")
        plt.show()
        
def main():
    '''
        main function
    '''
    #load model
    k_sparse = int(input("Enter k for training")) #K_sparse parameter that activates only k highest activation in z(hidden layer - bottleneck layer) and zeros out the rest 
    
    model = K_SparseAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr, betas=(0.9, 0.999), eps=1e-08)
    train_output = []
    test_output = []
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(num_epochs):
        #training
        
        train_loss = training(model,train_loader,train_output,epoch,optimizer,criterion,k_sparse)
        train_loss_list.append(train_loss)
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, num_epochs, train_loss))
        #testing
        
        test_loss = testing(model,test_loader,test_output,epoch,criterion,k_sparse)
        test_loss_list.append(test_loss)
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, num_epochs, test_loss))
    
    # plot the cost graph 
    cost_graph(train_loss_list,title="k-SAE k={} Train graph".format(k_sparse))
    cost_graph(test_loss_list,title="k-SAE k={} Test graph".format(k_sparse))
    print("training Images")

    #prints the output images 
    view_images(train_output,num_epochs,k_sparse)
    print("Testing Images")
    view_images(test_output,num_epochs,k_sparse)

    # saves the model 
    FILE = "K_SAE_model.pth"
    torch.save(model, FILE)
    
main()
