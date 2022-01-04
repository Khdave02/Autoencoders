# -*- coding: utf-8 -*-

import loader
import costGraph
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 1  # seed is used to ensure that we get the same output every time
torch.manual_seed(seed)

# set hyperparameter values
batch_size = 600
num_epochs = 10
alpha = 1e-3  # learning rate
lamda = 1e-4


class CAE_CNN(nn.Module):
    def __init__(self):
        super(CAE_CNN, self).__init__()
        # encoding layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)  # 28x28 --> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 14x14 --> 7x7
        self.conv3 = nn.Conv2d(64, 128, 5)  # 7x7 --> 3x3

        # decoding layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, 5)  # 3x3 --> 7x7
        self.tconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 7x7 --> 14x14
        self.tconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 14x14 --> 28x28

    # forward prop function
    def forward(self, x):
        # encoder
        # ret_val=[]
        # ret_weight = []
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = torch.sigmoid(self.tconv3(x))  # final layer is applied sigmoid activation

        return x


def contractive_loss(model):
    c_loss = 0
    for i in range(len(model.state_dict()) // 4):
        name = 'conv' + str(i + 1)
        w = model.state_dict()[name + '.weight']

        h = F.relu(w)
        dh = h * (1 - h)

        w_sum = torch.sum(w ** 2, dim=1)
        w_sum = w_sum.unsqueeze(1)
        c_loss += torch.sum((dh ** 2) * w_sum)
    return c_loss


def training(model, dataloader, opt, criterion, train_outputs, epoch):
    print('Training')
    # costs=[]
    running_loss = 0.0
    for (img, labels) in tqdm(dataloader):
        img = img.to(device)

        opt.zero_grad()
        # forward
        recon = model(img.float())

        mse_loss = criterion(recon, img)
        penalty = contractive_loss(model)
        loss = mse_loss + penalty * lamda

        # backward
        loss.backward()
        opt.step()

        running_loss += loss.item()

    train_outputs.append((epoch, img, recon))
    return running_loss / len(dataloader)


def testing(model, dataloader, criterion, test_op, epoch):
    print('Testing')
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (img, _) in tqdm(dataloader):
            # counter += 1
            img = img.to(device)
            recon = model(img).to(device)

            # mse_loss = criterion(recon, img)
            # penalty = contractive_loss(model)
            # loss = mse_loss + penalty * lamda
            loss = criterion(recon, img)

            running_loss += loss.item()

        test_op.append((epoch, img, recon))
    return running_loss / len(dataloader)


def view_images(outputs):
    for k in range(0, num_epochs, 3):
        plt.figure(figsize=(18, 5))
        plt.gray()
        og_imgs = outputs[k][1]

        og_imgs = og_imgs.cpu().detach().numpy()
        recon = outputs[k][2].cpu().detach().numpy()
        # print("Epoch: ",k)
        num = 10  # no. of images for a row

        # plots 10 input images and their reconstructed result
        plt.figure(figsize=(20, 5))
        plt.suptitle("Epoch: %i" % (k + 1))
        for i in range(num):
            ax = plt.subplot(2, num, i + 1)
            plt.imshow(og_imgs[i].reshape(28, 28), cmap="gray")
            plt.xticks([])
            plt.yticks([])  # removing axes
            ax.set_title('Original')

            ax = plt.subplot(2, num, i + 1 + num)
            plt.imshow(recon[i].reshape(28, 28), cmap="gray")
            # plotting the ith test image in subplot
            plt.xticks([])
            plt.yticks([])  # removing axes
            ax.set_title('Reconstructed')
        plt.show()


def main():
    train_loader = loader.trainLoader(batch_size)
    test_loader = loader.testLoader(batch_size)

    # load model
    model = CAE_CNN().to(device)

    params = model.parameters()
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(params, lr=alpha, betas=(0.9, 0.999), eps=1e-08)

    # training
    train_loss = []
    test_loss = []
    train_outputs = []
    test_op = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        train_epoch_loss = training(model, train_loader, opt, criterion, train_outputs, epoch)
        test_epoch_loss = testing(model, test_loader, criterion, test_op, epoch)
        print(f"Train Loss: {train_epoch_loss} \t Test Loss: {test_epoch_loss}")
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)

    costGraph.plot_cost(train_loss, test_loss, title="Contractive AE")

    # test the trained model on Test data
    view_images(test_op)

    FILE = 'CAE_model.pth'
    torch.save(model.state_dict(), FILE)
    # files.download('CAE_model.pth')


main()
