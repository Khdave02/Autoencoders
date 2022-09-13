
# K-Sparse Autoencoder 
k-Sparse Autoencoders finds the k highest activations in z (hidden layer) and zeros out the rest.
The error is then backpropogated only through the k active nodes in z.


Loss used:-
L1 Loss : This adds L1 Sparsity constraint to the activations of the neurons after the ReLU function, this adds sparsity effect to the weights. The hyperparameter λ controls how strong the penalty we want to apply on the sparsity loss. This is the penalty loss we add in addition to the loss function we use for the training of the network(such as BCELoss, MSELoss etc):
L1 Loss Function Image
![image](https://user-images.githubusercontent.com/87975841/189861258-1ca7943f-0503-46e1-8f03-91e5671af9b3.png)

The model contains:

- An encoder function g(.) parameterized by ϕ
- A decoder function f(.) parameterized by θ
- The low-dimensional code learned for input x in the bottleneck layer is the output of encoder, let's call it y
- The reconstructed input is z = gϕ(y)


The parameters (θ,ϕ) are learned together to output a reconstructed data sample same as the original input: $$

                              x' = fθ(gϕ(x))


Our target is to get:

                                  x' ≈ x
We have implemented the Sparse Autoencoder using PyTorch. You need to install these external libraries before running our code:

- pytorch(for model training)
- matplotlib(for plotting graphs and images)
- tqdm(for showing progress bars)
- numpy(for displaying images)

|Parameters     | values |
| ----------- | ----------- |
| Learning Rate      |     |
| Epochs  | Text        |
|Minibatch Size  | Text        |
|Optimizer  | Text        |
|Loss Function | Text        |
|Lambda  | Text        |

Input and Output Images 
train images
- k = 5

![](https://i.imgur.com/YadCixe.png)
- k = 10
![](https://i.imgur.com/qb2Rgfo.png)

- k = 25
![](https://i.imgur.com/DhnJBKt.png)

- k = 50
![](https://i.imgur.com/CSUCnpe.png)

- k = 100
![](https://i.imgur.com/gRfKETk.png)


test images 
- k = 5
![](https://i.imgur.com/f5okbws.png)
- k = 10
![](https://i.imgur.com/aJslcrC.png)
- k = 25
![](https://i.imgur.com/UfVVXuX.png)
- k = 50
![](https://i.imgur.com/xUCoe77.png)
- k = 100
![](https://i.imgur.com/JvGB7E9.png)
Loss Graph 

![](https://i.imgur.com/Safvdi3.png)
![](https://i.imgur.com/8seJ4Xf.png)


