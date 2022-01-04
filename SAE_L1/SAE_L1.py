import matplotlib.pyplot as plt  # for plotting images
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import loader
from tqdm import tqdm
seed = 1  # seed is used to ensure that we get the same output every time
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
train_loader = loader.trainLoader(100)
test_loader = loader.testLoader(600)
lr =1e-3
num_epochs=10
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 7)
        self.dec1 = nn.ConvTranspose2d(64, 32, 7)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        # encoding
        ret_val=[]
        x = F.relu(self.enc1(x))
        ret_val.append(x)

        x = F.relu(self.enc2(x))
        ret_val.append(x)
        x = F.relu(self.enc3(x))
        ret_val.append(x)

        x = F.relu(self.dec1(x))
        ret_val.append(x)
        x = F.relu(self.dec2(x))
        ret_val.append(x)
        x = F.relu(self.dec3(x))
        ret_val.append(x)
        return x,ret_val

def sparse_loss(model_children):
    loss = 0
    lamda=1e-4
    for i in model_children:
        loss += torch.mean(torch.abs(i))
    return lamda*loss
def training(model,trainloader,train_output,epoch,optimizer,criterion):
    print("Training")
    loss = 0.0                  # Looping for every epoch
    for img,_ in tqdm(trainloader):  # Looping for every batch
            optimizer.zero_grad()  # Model training starts here
            outputs,model_children=model(img)
            train_loss = criterion(outputs, img) + sparse_loss(model_children)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
    loss = loss / len(trainloader)
    train_output.append((epoch, img, outputs))
    return loss
def testing(model,testloader,test_output,epoch,criterion):
    print("Testing")
    with torch.no_grad():
            loss = 0.0
            for img,_ in tqdm(testloader):
                img = img.to(device)
                outputs,model_children=model(img.float())
                loss = criterion(outputs, img) #+ sparse_loss(model_children)
                loss += loss.item()
            loss = loss / len(testloader)
            test_output.append((epoch, img, outputs))
    return loss
def cost_graph(loss_list,title):
    plt.plot(loss_list)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title(title)
    plt.show()
 
def view_images(output,num_epochs):
    for k in range(0,num_epochs):
        plt.figure(figsize=(18, 5))
        plt.gray()
        ori_imgs = output[k][1]
        recon = output[k][2]
        n= 9
        for i in range(n):
            plt.subplot(2, n, i+1)
            plt.title("Original")
            plt.imshow(ori_imgs[i].detach().numpy().reshape(28,28),cmap = "gray")
            plt.subplot(2, n, i+1+n)
            plt.title("Sparse")
            plt.imshow(recon[i].detach().numpy().reshape(28,28),cmap = "gray")
        plt.show()
        
def main():
    
    #load model
    model = SparseAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr, betas=(0.9, 0.999), eps=1e-08)
    train_output = []
    test_output = []
    train_loss_list = []
    test_loss_list = []
    #training
    for epoch in range(num_epochs):
        train_loss = training(model,train_loader,train_output,epoch,optimizer,criterion)
        train_loss_list.append(train_loss)
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, num_epochs, train_loss))

        test_loss = testing(model,test_loader,test_output,epoch,criterion)
        test_loss_list.append(test_loss)
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, num_epochs, test_loss))

    cost_graph(train_loss_list,title="CAE Train graph")
    view_images(train_output,num_epochs)
    #testing
    cost_graph(test_loss_list,title="CAE Test graph")
    view_images(test_output,num_epochs)
    FILE = "SAE_L1_model.pth"
    torch.save(model, FILE)
    
main()