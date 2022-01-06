# Contractive Autoencoder
Contractive Autoencoder is another unsupervised deep learning technique that makes an autoencoder robust of small changes to the training dataset.

To achieve this goal, a penalty term is added to the loss function that penalizes representations being too sensitive towards the training input data. The sensitivity is measured by the Frobenius norm of the Jacobian matrix of the encoder activations with respect to the input:
-img
Hence the loss is as follows:
-img

---
## Hyperparameters used
| Hyperparameter |value          |
| ------------- | ------------- |
| Batch-Size    | 600           |
| Learning-rate | 0.001       |
| Lambda       |  0.0001
| no. of Epochs | 10            |
|  Loss         |  MSE Loss     |
|  Optimizer    | Adam Optimizer|


---
## Model Architecture
``` python
        # encoding layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)  # 28x28 --> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 14x14 --> 7x7
        self.conv3 = nn.Conv2d(64, 128, 5)  # 7x7 --> 3x3

        # decoding layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, 5)  # 3x3 --> 7x7
        self.tconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 7x7 --> 14x14
        self.tconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 14x14 --> 28x28
```
## Result and Outputs
### Loss vs No. of Epochs graph
-img

After 10 epochs, The training and testing losses were:

| Type | Loss value | 
| -------- | -------- | 
| Train loss     |  0.00498754    |
| Test Loss      | 0.00216635     |


### Outputs 
-img


---



