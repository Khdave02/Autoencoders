# Sparse_kl_div-Autoencoder
Implementation of Sparse_kldiv Autoencoder on MNIST by using Pytorch and CNN

In sparse autoencoder training criterion involves a sparsity penalty which would construct loss function by penalizing activations of hidden layers so that only a few nodes are encouraged to activate when a single sample is fed into the network.



## MNIST-Dataset
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is used. The dataset contains 60,000 training samples and 10,000 test samples. Each sample in the MNIST dataset is a 28x28 pixel grayscale image of a single handwritten digit between 0 & 9.

 

![mnistDataset](https://user-images.githubusercontent.com/83291620/148073577-ac3e4e3d-382a-4616-be4e-48e72a0aaf88.png)

## kl_div_loss
The average of the activations of the jth neuron is calculated as

![image](https://user-images.githubusercontent.com/87975841/148243235-5c8294ef-30fa-4f80-84a4-cf8db5ed1969.png)

We will add another sparsity penalty in terms of ρ^j and ρ to  MSELoss. The penalty will be applied on ρ^j when it will deviate too much from ρ. 
The following is the formula for the sparsity penalty.

![image](https://user-images.githubusercontent.com/87975841/148243287-466a7d2f-9413-4b68-8900-f91d1dae82a2.png)


where s is the number of neurons in the hidden layer.







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
        self.enc1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 7)
        self.dec1 = nn.ConvTranspose2d(64, 32, 7)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
 ```

## Loss plots of Epochs
![Screenshot 2022-01-05 213250](https://user-images.githubusercontent.com/87975841/148249131-44c25179-8506-4ac9-a8b0-1239bce1733b.png)


## Results and Outputs
![train_output](https://user-images.githubusercontent.com/87975841/148247639-0449fe2c-634f-4ce4-9f05-a287981fa215.png)




