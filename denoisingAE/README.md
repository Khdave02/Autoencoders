# MNIST-Image-Denoising-Autoencoder-Using-CNN
Implementation of Image-Denoising Autoencoder on MNIST by using Pytorch and CNN




## MNIST-Dataset
The [MNIST](http://yann.lecun.com/exdb/mnist/) and [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) datasets are used. These datasets contain 60,000 training samples and 10,000 test samples. Each sample in the MNIST dataset is a 28x28 pixel grayscale image of a single handwritten digit between 0 & 9, whereas each sample in the Fashion MNIST dataset is a 28x28 grayscale image associated with a label from 10 types of clothing.

 

![](https://i.imgur.com/FhAVzAp.png)









## Hyperparameters

| Hyperparaeter |value          |
| ------------- | ------------- |
| Batch-Size    | 64            |
| Learning-rate | 0.001         |
| Weight-decay  | 0.00001       |
| num of Epochs | 10            |
|  Loss         |  MSE LOSS     |
|  Optimizer    | Adam Optimizer|

## Adding Noise
* ```torch.randn``` is used to create a noisy tensor of the same size as the input. The amount of Gaussian noise can be changed by changing the multiplication factor.
 


## Architecture 

```   python
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
      
        
 ```
## Training with flowchart


In denoising autoencoder some noise is introduced to the input images. The encoder network downsamples the data into a lower-dimensional latent space and then the decoder reconstructs the original data from the lower-dimensional representation. MSE loss between the original image and the reconstructed image is calculated and is backpropagated. Value of the parameters is updated using Adam optimization to reduce the reconstruction error.

## Loss plots of Epochs
![Screenshot 2022-01-04 191407](https://user-images.githubusercontent.com/87975841/148068236-750c4830-767d-467f-b770-75487c785dc0.png)


## Results and Outputs
![Screenshot 2022-01-04 191527](https://user-images.githubusercontent.com/87975841/148068286-963d691c-9c8b-4f2a-a3af-5f07c508778e.png)



