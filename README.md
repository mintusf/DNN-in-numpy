# DNN-in-numpy

Implementation of DNN algorithm with using only the numpy library. Model is trained using batch gradient descent and tested on MNIST dataset achieving 94% accuracy on test set.
Dropout and L2 are implemented as regularization algorithms. The model is fully scallable, which means that regularization parameter as well as numbers of hidden layers and nodes can be set to any value. 
All functions, including forward propagation, back propagation, cross entropy loss calculation, dropout and training algorithm are written without the tensorflow library. 
Different activation function can be used: sigmoid, Relu or hyperbolic tangent.

 
Hyperparameters:
-> layers_size - vector containing numbers of nodes in hidden layers 
    (any number of layers and nodes is possible)
-> reg_lambda - the regularization parameter lambda
-> activ_func - the activation function used in hidden layers 
    (in output layer, always sigmoid function is used). 
    Possible choices are sigmoid, Relu and tanh.
-> learning_rate
-> drop_prob - the probability of supressing a node in the dropout algorithm

NN.train is a function which trains the model. Hyperparameters are:
-> steps_no - number of epochs

NN.pred_val is a function which predicts labels for any input and shows random images. 
-> im_to_show - number of images to show with predictions
