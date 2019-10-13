# DNN-in-numpy

Implementation of DNN algorithm with using only the numpy library. Model is trained using batch gradient descent and tested on MNIST dataset achieving 94% accuracy on test set.
The model is fully scallable, which means that regularization parameter as well as numbers of hidden layers and nodes can be set to any value. 
All functions, including forward propagation, back propagation, cross entropy loss calculation and training algorithm are written without the tensorflow library. 
Different activation function can be used: sigmoid, Relu or hyperbolic tangent.
