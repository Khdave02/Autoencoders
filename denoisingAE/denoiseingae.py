# -*- coding: utf-8 -*-

import loader  # module for loading dataset and dataloader
import costGraph  # module to plot cost graph of train & test losses
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm  # for showing progress bars during training and testing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

FILE = 'denoising_model.pth'  # Path where the model is saved/loaded

seed = 1  # seed is used to ensure that we get the same output every time
torch.manual_seed(seed)

# set the hyperparameters values
batch_size = 600
num_epochs = 10
alpha = 5e-3  # learning rate
noise_f = 0.4  # noise factor to add gaussian noise


def add_noise(img, noise_f):
    '''
    This function adds random gaussian noise to the input image
    :param img: input image
    :param noise_f: noise factor
    Returns the noised image on adding gaussian noise
    '''
    noise = torch.randn(img.size()) * noise_f  # Added gaussian noise
    noisy_img = img + noise
    return noisy_img


class AE_CNN(nn.Module):
    '''
        CNN model for the autoencoder
    '''

    def __init__(self):
        super(AE_CNN, self).__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # -> N, 64, 1, 1
        )

        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()  # img pixel values ranges from 0-1 so sigmoid used for final activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def training(model, dataloader, opt, criterion, train_outputs, epoch):
    '''
    This function trains the Neural Network.
    Parameters-
    model: CNN AutoEncoder model
    dataloader: DataLoader object that iterates through the train set
    opt: optimizer object to be used for training
    criterion: MSE loss function to be used for training
    train_outputs: list containing per epoch results(input & output) obtained after training
    epoch: current epoch
    Returns: train loss per epoch
    '''
    print('Training')
    running_loss = 0.0
    for (img, labels) in tqdm(dataloader):
        noisy_img = add_noise(img, noise_f)
        noisy_img = noisy_img.to(device)
        opt.zero_grad()
        # forward
        reconstructed_img = model(noisy_img).to(device)

        loss = criterion(reconstructed_img, img.to(device))
        running_loss += loss.item()
        # backward
        loss.backward()
        opt.step()

    train_outputs.append((epoch, img, reconstructed_img))
    return running_loss / len(dataloader)


def testing(model, dataloader, criterion, test_op, epoch):
    '''
        This function tests the trained model
        Parameters-
        model: CNN AutoEncoder model
        dataloader: DataLoader object that iterates through the test set
        criterion: MSE loss function to be used for testing
        test_outputs: list containing per epoch results(input & output) obtained after testing
        epoch: current epoch
        Returns: test loss per epoch
    '''
    print('Testing')
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader):
            img, _ = data
            noisy_img = add_noise(img.float(), noise_f)
            noisy_img = noisy_img.to(device)
            recon = model(noisy_img).to(device)
            loss = criterion(recon, img.to(device))
            running_loss += loss.item()

        test_op.append((epoch, img, recon))
    return running_loss / len(dataloader)


def view_images(outputs):
    '''
    This function displays the result images
    :param outputs: list containing per epoch results(input & output) obtained after training/testing
    '''
    for k in range(0, num_epochs, 3):  # printing results after every 3 epochs
        plt.figure(figsize=(18, 5))
        plt.gray()
        og_imgs = outputs[k][1]
        noisy_imgs = add_noise(og_imgs, noise_f)  # added gaussian noise

        noisy_imgs = noisy_imgs.cpu().detach().numpy()
        og_imgs = og_imgs.detach().numpy()
        recon = outputs[k][2].cpu().detach().numpy()

        num = 10  # no. of images in a row

        # plot the original images first
        plt.figure(figsize=(18, 5))
        plt.suptitle("Epoch: %i" % (k + 1))
        for i in range(num):
            plt.subplot(2, num, i + 1)
            plt.title("Original")
            plt.imshow(og_imgs[i].reshape(28, 28), cmap="gray")
        plt.show()

        # plots 10 input noisy images and their reconstructed result
        plt.figure(figsize=(18, 5))
        plt.suptitle("Epoch: %i" % (k + 1))
        for i in range(num):
            ax = plt.subplot(2, num, i + 1)
            plt.imshow(noisy_imgs[i].reshape(28, 28), cmap="gray")
            plt.xticks([])
            plt.yticks([])  # removing axes
            ax.set_title('Noisy')

            ax = plt.subplot(2, num, i + 1 + num)
            plt.imshow(recon[i].reshape(28, 28), cmap="gray")
            plt.xticks([])
            plt.yticks([])  # removing axes
            ax.set_title('Denoised')
        plt.show()


def main():
    '''
    This is the main function of the code
    '''
    train_loader = loader.trainLoader(batch_size)
    test_loader = loader.testLoader(batch_size)

    model = AE_CNN().to(device)
    params = model.parameters()
    criterion = nn.MSELoss()

    opt = torch.optim.Adam(params, lr=alpha, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    train_loss = []
    test_loss = []
    train_outputs = []
    test_op = []

    ch = input("Press l to load model, t to train model: ").lower()  # asks user if they want to train the model or load the already saved model
    if ch == 'l':
        model.load_state_dict(torch.load(FILE))  # loads the model
        # test the loaded model on the test data
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            test_epoch_loss = testing(model, test_loader, criterion, test_op, epoch)
            print(f"Test Loss: {test_epoch_loss}")
            test_loss.append(test_epoch_loss)

    elif ch == 't':
        # training and testing
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            train_epoch_loss = training(model, train_loader, opt, criterion, train_outputs, epoch)
            test_epoch_loss = testing(model, test_loader, criterion, test_op, epoch)
            print(f"Train Loss: {train_epoch_loss} \t Test Loss: {test_epoch_loss}")
            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)
        # plot the cost graph
        costGraph.plot_cost(train_loss, test_loss, title="Denoising AE")
        torch.save(model.state_dict(), FILE)  # saves the trained model at the specified path

    # view the output images obtained after testing
    view_images(test_op)


main()
