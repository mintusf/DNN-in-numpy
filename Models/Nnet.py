#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def init():  # function to import mnist dataset
    mnist = input_data.read_data_sets('MNIST/', one_hot=False)
    rand_indx_train = np.random.randint(0, 54999, 5000)
    X_train = mnist.train.images[rand_indx_train]
    y_train = mnist.train.labels[rand_indx_train]
    X_test = mnist.test.images
    y_test = mnist.test.labels
    return (X_train, y_train, X_test, y_test)

# Activation functions

def sigmoid(z, derivative=False):
    out = 1. / (1. + np.exp(-z))
    if derivative:
        g = np.multiply(out, 1. - out)
        return np.array(g)
    return np.array(out)


def relu(z, derivative=False):
    if derivative:
        return np.array(1. * (z > 0))
    return np.array(1. * (z > 0) * z)


def tanh(z, derivative=False):
    out = np.tanh(z)
    if derivative:
        return np.array(1 - out * out)
    return np.array(out)

# uniform He initialization of weights

def randomW(shape):  
    epsilon = np.sqrt(2) / np.sqrt(shape[1])
    return np.random.uniform(-epsilon, epsilon, shape)

# Class definition for a layer

class Layer:  # Layer class

    def __init__(
        self,
        node_no,
        activ_func,
        lay_no,
        lay_type,
        sets_no,
        ):
        self.node_no = node_no
        self.lay_no = lay_no
        self.type = lay_type
        self.activ_func = activ_func
        self.val_in = np.zeros((sets_no, self.node_no))
        self.delta = np.zeros((sets_no, self.node_no))
        self.val_out = activ_func(self.val_in)
        self.val_out_withdropout = activ_func(self.val_in)
        self.dropout = np.zeros((sets_no, self.node_no))

    def feed(self, data):
        self.val_in = data

    def activate(self):
        self.val_out = self.activ_func(self.val_in)

# Class definition for a weights matrix
        
class Weights:

    def __init__(self, shape, w_no):
        self.val = randomW(shape)
        self.w_no = w_no
        self.grad = np.zeros(shape)
        self.weighted_grad = np.zeros(shape)
        self.weighted_grad_corrected = np.zeros(shape)
        self.weighted_square = np.zeros(shape)
        self.weighted_square_corrected = np.zeros(shape)
        self.shape = shape

    def grad_zero(self):
        self.grad = np.zeros(self.shape)

    def update(self, l_rate,optimizer,epsilon = 10**(-9)):
        if optimizer=="Grad_desc":
            self.val -= l_rate * self.grad
            self.grad_zero()
        if optimizer =="Adam":
            self.val -= l_rate * (self.weighted_grad_corrected/np.sqrt(self.weighted_square_corrected + epsilon))
            self.grad_zero()

# Class definition for a DNN model
        
class NNet:  # DNN model class

    def __init__(
        self,
        input_lay_size,
        output_lay_size,
        layers_size,
        reg_lambda,
        drop_prob,
        activ_func,
        X,
        y,
        learning_rate,
        beta1,
        beta2,
        optimizer,
        ):
        self.X_data = np.concatenate((np.ones((X.shape[0], 1)), X),
                axis=1)
        self.y = y
        self.reg_lambda = reg_lambda  # regularization parameter lambda
        self.drop_prob = drop_prob
        self.learning_rate = learning_rate
        self.layers = []
        self.sets_no = self.X_data.shape[0]
        self.beta1 = beta1
        self.beta2 = beta2
        self.optimizer = optimizer
        self.iteration = 1


        # initialization of layers
        # input layer

        self.layers.append(Layer(input_lay_size, lambda x: x,
                           lay_type='Input', lay_no=0,
                           sets_no=self.sets_no))
        self.layers[0].feed(X)

        # hidden layers

        for lay in range(len(layers_size)):
            self.layers.append(Layer(layers_size[lay], activ_func,
                               lay_type='Hidden', lay_no=lay + 1,
                               sets_no=self.sets_no))

        # output layer
        
        self.layers.append(Layer(output_lay_size, sigmoid,
                           lay_type='Output', lay_no=len(layers_size)
                           + 1, sets_no=self.sets_no))

        # initialization of weights

        self.weights = []
        self.weights_grad = []
        for lay in range(len(layers_size) + 1):
            self.weights.append(Weights((self.layers[lay + 1].node_no,
                                self.layers[lay].node_no + 1), lay))
            
    def forward_prop(self, dropout = True):  # function calculating activations of all layers & model's output
        for layer in self.layers:
            if layer.type == 'Input':
                layer.activate()
                layer.val_out_withdropout = layer.val_out
            else:
                
                # Dropout implementation
                
                if layer.type == 'Hidden' and dropout:
                    
                    layer.dropout = np.random.randn(self.sets_no, layer.node_no)
                    layer.dropout = (layer.dropout <= self.drop_prob).astype(int)
                    layer.val_out_withdropout = layer.val_out * layer.dropout
                    layer.val_out_withdropout = layer.val_out_withdropout / self.drop_prob
                    
                layer.val_in = \
                    np.matmul(np.concatenate((np.ones((self.X_data.shape[0],
                              1)), self.layers[layer.lay_no
                              - 1].val_out_withdropout), axis=1),
                              self.weights[layer.lay_no
                              - 1].val.transpose())
                layer.activate()
                
    # calculating cross-entropy cost function with regularization
    
    def calc_lossfcn(self, input_y):
        cost1 = np.sum(np.diag(-1 / input_y.shape[0]
                       * np.matmul(np.log(self.layers[-1].val_out),
                       input_y.transpose())))
        cost2 = np.sum(np.diag(-1 / self.sets_no * np.matmul(np.log(1
                       - self.layers[-1].val_out), (1
                       - input_y).transpose())))
        cost_reg = 0
        for W in self.weights:
            cost_reg += float(np.matmul(W.val[:, 1:].reshape((-1,
                              1)).transpose(), W.val[:, 1:
                              ].reshape((-1, 1))))
        cost_reg *= self.reg_lambda / 2 / self.sets_no
        self.cost = cost1 + cost2 + cost_reg
        return self.cost            
 
     # function executing back propagation to calculate derivatives of 
     # lost function with respect to all layers' activations

    def back_prop(self): 
        for layer in reversed(self.layers):
            if layer.type == 'Output':
                layer.delta = layer.val_out - self.y
            elif layer.type == 'Input':
                pass
            elif self.layers[layer.lay_no + 1].type == 'Output':
                layer.delta = \
                    np.multiply(np.matmul(self.layers[layer.lay_no
                                + 1].delta,
                                self.weights[layer.lay_no].val),
                                np.concatenate((np.zeros((self.sets_no,
                                1)), sigmoid(layer.val_in,derivative = True)),
                                axis=1))
            else:
                layer.delta = \
                    np.multiply(np.matmul(self.layers[layer.lay_no
                                + 1].delta[:, 1:],
                                self.weights[layer.lay_no].val),
                                np.concatenate((np.zeros((self.sets_no,
                                1)), sigmoid(layer.val_in,derivative = True)),
                                axis=1))
            if layer.type == 'Hidden':
                    layer.dropout = np.concatenate ((np.ones((self.X_data.shape[0],1)),  layer.dropout),axis = 1)
                    layer.delta = layer.delta * layer.dropout
                    layer.delta = layer.delta / self.drop_prob
                
    # A function updating weights according to calculated gradient with dropout
          
    def weights_update (self):
        for W in self.weights:
            W.grad_zero()
            if self.layers[W.w_no + 1].type != 'Output':              
                W.grad = np.dot(self.layers[W.w_no
                            + 1].delta[:, 1:].T,
                            np.concatenate((np.ones((self.sets_no, 1)),
                            self.layers[W.w_no].val_out_withdropout),axis = 1))
            else:
                W.grad = np.dot(self.layers[W.w_no
                            + 1].delta.T,
                            np.concatenate((np.ones((self.sets_no, 1)),
                            self.layers[W.w_no].val_out_withdropout),axis = 1))

            # regularizated update

            W.grad[:, 1:] += self.reg_lambda * W.val[:, 1:]
            W.grad /= self.sets_no
            
            # Implementation of Adam optimizer

            if self.optimizer == "Adam":
                
                # Momentum calculation
                
                W.weighted_grad = self.beta1 * W.weighted_grad + (1 - self.beta1) * W.grad
                W.weighted_grad_corrected = W.weighted_grad / (1 - self.beta1 ** self.iteration)
                
                # RMSprop calculation
                
                W.weighted_square = self.beta2 * W.weighted_square + (1 - self.beta2) * (W.grad ** 2)
                W.weighted_square_corrected = W.weighted_square / (1 - self.beta2 ** self.iteration)
                    
            W.update(self.learning_rate, self.optimizer)


    def train(self):
        self.forward_prop()
        self.back_prop()
        self.weights_update()
        self.iteration += 1

    # function used for prediction using learnt model
        
    def predict(self, input_X, input_y):
        ins = []
        outs = []
        for layer in self.layers:
            ins.append(layer.val_in)
            outs.append(layer.val_out)
            layer.val_in = np.zeros((input_X.shape[0], layer.node_no))
            layer.feed(input_X)
            if layer.type == 'Input':
                layer.val_in = input_X
                layer.activate()
            else:
                layer.val_in = \
                    np.matmul(np.concatenate((np.ones((input_X.shape[0],
                              1)), self.layers[layer.lay_no
                              - 1].val_out), axis=1),
                              self.weights[layer.lay_no
                              - 1].val.transpose())
                layer.activate()
        predictions = self.layers[-1].val_out.argmax(axis=1) % 10
        cost = self.calc_lossfcn(input_y)
        for layer in self.layers:
            layer.val_in = np.zeros((self.sets_no, layer.node_no))
            layer.val_in = ins[layer.lay_no]
            layer.val_out = np.zeros((self.sets_no, layer.node_no))
            layer.val_out = outs[layer.lay_no]
        return (predictions, cost)
 

# function for training and visualizing performance    
    
def train(  
    model,
    steps_no,
    X_train,
    y_train,
    X_test,
    y_test,
    ):
    steps = np.linspace(1, steps_no, steps_no)
    loss_train_axis = []
    loss_test_axis = []
    acc_train_axis = []
    acc_test_axis = []
    iter_axis = []
    labels = np.array([i for i in range(10)])
    y_train_formodel = np.array([y_train[i] == labels for i in
                                range(y_train.shape[0])])
    y_test_formodel = np.array([y_test[i] == labels for i in
                               range(y_test.shape[0])])
    for i in steps:
        model.train()
        print(f"                             {model.calc_lossfcn(y_train_formodel)}")
        if i % 10 == 0:
            iter_axis.append(i)
            [predictions_train, loss_train] = model.predict(X_train,
                    y_train_formodel)
            [predictions_test, loss_test] = model.predict(X_test,
                    y_test_formodel)
            loss_train_axis.append(loss_train)
            loss_test_axis.append(loss_test)
            acc_train_axis.append(np.mean(y_train == predictions_train))
            acc_test_axis.append(np.mean(y_test == predictions_test))
            print(f"TRAIN: {loss_train}__TEST: {loss_test}")
            print(f"Train accuracy: {np.mean(y_train==predictions_train)}     Test accuracy: {np.mean(y_test==predictions_test)}")
            plt.plot(iter_axis,loss_train_axis)
            plt.plot(iter_axis,loss_test_axis)
            plt.show()
    if i>10:
        print(f"Final accuracy: {np.mean(y_test==predictions_test)}")
    return (loss_train_axis, loss_test_axis, acc_train_axis,
            acc_test_axis)
     
def pred_val(  # function validating learnt model by recognizing digits from random images
    model,
    imag_no,
    X_test,
    y_test,
    ):
    rand_indx = np.random.randint(0, 4999, imag_no)
    labels = np.array([i for i in range(10)])
    y_test_formodel = np.array([y_test[i] == labels for i in
                               range(y_test.shape[0])])
    (results, loss) = model.predict(X_test[rand_indx],
                                    y_test_formodel[rand_indx])
    for i in range(imag_no):
        plt.imshow(X_test[rand_indx[i], :].reshape((28, 28)))
        plt.show()
        print(f"True label: {int((y_test[rand_indx[i]])%10)}__ predicted: {results[i]}")

