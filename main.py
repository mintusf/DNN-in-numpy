#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import Models.Nnet as NN

if __name__ == '__main__':
    (X_train, y_train, X_test, y_test) = NN.init()
    labels = np.array([i for i in range(10)])
    y_train_formodel = np.array([y_train[i] == labels for i in
                                range(y_train.shape[0])])
    y_test_formodel = np.array([y_test[i] == labels for i in
                               range(y_test.shape[0])])
    
    """
    Model is an instance of the NN class. Hyperparameters:
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
    """
    
    steps_no = 50
    Model = NN.NNet(
        input_lay_size=X_train.shape[1],
        output_lay_size=y_train_formodel.shape[1],
        layers_size=[120,120],
        reg_lambda=0,
        drop_prob = 0.9,
        activ_func=NN.tanh,
        X=X_train,
        y=y_train_formodel,
        learning_rate = 0.5,
        )
    (loss_train, loss_test, acc_train, acc_test) = NN.train(
        Model, steps_no, X_train, y_train, X_test, y_test)
    im_to_show = 10
    NN.pred_val(Model, im_to_show, X_test, y_test)
