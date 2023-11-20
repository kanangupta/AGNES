# Overview
This repository contains a Pytorch implementation of the AGNES optimization algorithm and as described in [Kanan Gupta, Jonathan Siegel, and Stephan Wojtowytsch. "Achieving acceleration despite very noisy gradients." arXiv preprint arXiv:2302.05515 (2023).](https://arxiv.org/pdf/2302.05515.pdf) as well as the code for the experiments reported therein.

## Using AGNES
`AGNES.py` contains the implementation of the aglorithm using the [Pytorch optimizer class](https://pytorch.org/docs/stable/optim.html). To use the algorithm, simply import `AGNES.py` and initialize the optimizer with model parameters that you wish to optimize and hyperparameters of your choice.
```
import AGNES
opt = AGNES(model.parameters(), lr=1e-3 , momentum=0.99, correction=1e-2)
```
If you do not specify hyperparameters, the default values recommended in the paper are used.

## Experiments
The two Jupyter notebooks implement the algorithm for some convex and strongly convex functions and compare its performance to SGD and Nesterov's accelerated gradient descent. `mnist_grid_search.py` compares different combinations of the hyperparameters alpha and eta for training LeNet-5 on the MNIST dataset and `grid_search_plot.py` plots the results obtained.

`nn_experiments.py` is used for training neural network models on image datasets (MNIST, CIFAR-10, and CIFAR-100) using various optimization algorithms, including AGNES. The code for neural network experiments is based on Matthias Wright's code available at https://github.com/matthias-wright/cifar10-resnet. `plots.py` plots the results obtained from the neural network experiments.
See the paper for a description and results of the experiments.
