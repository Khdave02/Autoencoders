# Sparse_L1-Autoencoder
Implementation of Sparse_L1 Autoencoder on MNIST by using Pytorch and CNN

In sparse autoencoder training criterion involves a sparsity penalty which would construct loss function by penalizing activations of hidden layers so that only a few nodes are encouraged to activate when a single sample is fed into the network.



## MNIST-Dataset
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is used. The dataset contains 60,000 training samples and 10,000 test samples. Each sample in the MNIST dataset is a 28x28 pixel grayscale image of a single handwritten digit between 0 & 9.

 

![mnistDataset](https://user-images.githubusercontent.com/83291620/148073577-ac3e4e3d-382a-4616-be4e-48e72a0aaf88.png)

## L1_regularization_loss
L1 sparsity constraint is added to the activations of the neuron after the ReLU function. This will make some of the weights to be zero which will add a sparsity effect to the weights

![Screenshot 2022-01-06 185917](https://user-images.githubusercontent.com/87975841/148390492-43342f5b-8bb5-429d-bdd1-6a06b2bf7e2f.png)

Here, λ is the regularization parameter, and wis are the activation weights. We will add this regularization to the loss function i.e MSELoss.
![Screenshot 2022-01-06 183047](https://user-images.githubusercontent.com/87975841/148390451-d9bcdb02-3861-4df9-9e21-b05536a3d9b3.png)

## Hyperparameters

| Hyperparaeter  |value          |           
| -------------  | ------------- |     
|Train-Batch-Size| 100           |
|Test-Batch-Size | 600           |
| Learning-rate  | 0.001         |
| num of Epochs  | 10            |
|  Loss          | MSE LOSS      |
|  Optimizer     | Adam Optimizer|
|  beta          | 0.005         |
|  rho           | 0.01          |


## Architecture 

```   python
        # Encoding 
        self.enc1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 7)
        # Decoding
        self.dec1 = nn.ConvTranspose2d(64, 32, 7)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
 ```

## Loss plots of Epochs
![Screenshot 2022-01-06 192108](https://user-images.githubusercontent.com/87975841/148393108-14b71d11-110f-4d31-9107-99f6ca2de0e8.png)


## Results and Outputs

   |Type            |Loss           |           
   | -------------  | ------------- |     
   |Training Loss   | 0.00487120    |
   |Testing Loss    | 0.00054854    |
   
   
Results on Test Set:-
![test_outpt](https://user-images.githubusercontent.com/87975841/148391659-5455561c-9879-4331-9af5-2aaf0c29f7e9.png)




