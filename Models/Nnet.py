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

def sigmoid(z):
    sigm = 1. / (1. + np.exp(-z))
    return np.array(sigm)


def sigmoidGradient(z):
    g = np.multiply(sigmoid(z), 1. - sigmoid(z))
    return np.array(g)


def randomW(shape):  # uniform Xavier initialization of weights
    epsilon = np.sqrt(6) / np.sqrt(shape[0] + shape[1])
    return np.random.uniform(-epsilon, epsilon, shape)

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

    def feed(self, data):
        self.val_in = data

    def activate(self):
        self.val_out = self.activ_func(self.val_in)

        
class Weights:  # Layer class

    def __init__(self, shape, w_no):
        self.val = randomW(shape)
        self.w_no = w_no
        self.grad = np.zeros(shape)
        self.shape = shape

    def grad_zero(self):
        self.grad = np.zeros(self.shape)

    def update(self, l_rate):
        self.val -= l_rate * self.grad
        self.grad_zero


class NNet:  # DNN model class

    def __init__(
        self,
        input_lay_size,
        output_lay_size,
        layers_size,
        reg_lambda,
        activ_func,
        X,
        y,
        learning_rate,
        ):
        self.X_data = np.concatenate((np.ones((X.shape[0], 1)), X),
                axis=1)
        self.y = y
        self.reg_lambda = reg_lambda  # regularization parameter lambda
        self.learning_rate = learning_rate
        self.layers = []
        self.sets_no = self.X_data.shape[0]

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
        
        self.layers.append(Layer(output_lay_size, activ_func,
                           lay_type='Output', lay_no=len(layers_size)
                           + 1, sets_no=self.sets_no))

        # initialization of weights

        self.weights = []
        self.weights_grad = []
        for lay in range(len(layers_size) + 1):
            self.weights.append(Weights((self.layers[lay + 1].node_no,
                                self.layers[lay].node_no + 1), lay))
            
    def forward_prop(self):  # function calculating activations of all layers & model's output
        for layer in self.layers:
            if layer.type == 'Input':
                layer.activate()
            else:
                layer.val_in = \
                    np.matmul(np.concatenate((np.ones((self.X_data.shape[0],
                              1)), self.layers[layer.lay_no
                              - 1].val_out), axis=1),
                              self.weights[layer.lay_no
                              - 1].val.transpose())
                layer.activate()
                
        # calculating cross-entropy cost function with regularization

        cost1 = np.sum(np.diag(-1 / self.sets_no
                       * np.matmul(np.log(self.layers[-1].val_out),
                       self.y.transpose())))
        cost2 = np.sum(np.diag(-1 / self.sets_no * np.matmul(np.log(1
                       - self.layers[-1].val_out), (1
                       - self.y).transpose())))
        cost_reg = 0
        for W in self.weights:
            cost_reg += float(np.matmul(W.val[:, 1:].reshape((-1,
                              1)).transpose(), W.val[:, 1:
                              ].reshape((-1, 1))))
        cost_reg *= self.reg_lambda / 2 / self.sets_no
        self.cost = cost1 + cost2 + cost_reg
        return self.cost
    
    def calc_lossfcn(self, input_y):
        cost1 = np.sum(np.diag(-1 / self.sets_no
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
    
    def back_prop(self):  # function executing back propagation to calculate weights gradient
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
                                1)), sigmoidGradient(layer.val_in)),
                                axis=1))
            else:
                layer.delta = \
                    np.multiply(np.matmul(self.layers[layer.lay_no
                                + 1].delta[:, 1:],
                                self.weights[layer.lay_no].val),
                                np.concatenate((np.zeros((self.sets_no,
                                1)), sigmoidGradient(layer.val_in)),
                                axis=1))
    def train(self):
        self.forward_prop()
        self.back_prop()
        for W in self.weights:
            W.grad_zero()
            if self.layers[W.w_no + 1].type != 'Output':
                for i in range(5000):
                    W.grad = W.grad + np.dot(self.layers[W.w_no
                            + 1].delta[i, 1:][:, None],
                            np.concatenate((np.ones((1, 1)),
                            self.layers[W.w_no].val_out[i, :][None, :
                            ]), axis=1))
            else:

                for i in range(5000):
                    W.grad = W.grad + np.dot(self.layers[W.w_no
                            + 1].delta[i, :][:, None],
                            np.concatenate((np.ones((1, 1)),
                            self.layers[W.w_no].val_out[i, :][None, :
                            ]), axis=1))

            # regularizated update

            W.grad[:, 1:] += self.reg_lambda * W.val[:, 1:]
            W.grad /= self.sets_no
            W.update(self.learning_rate)
        
    def predict(self, input_X, input_y):  # function used for prediction using learnt model
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
 
    
def train(  # function for training and visualizing performance
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
    labels = np.array([i for i in range(10)])
    y_train_formodel = np.array([y_train[i] == labels for i in
                                range(y_train.shape[0])])
    y_test_formodel = np.array([y_test[i] == labels for i in
                               range(y_test.shape[0])])
    for i in steps:
        model.train()
        print(f"                             {model.calc_lossfcn(y_train_formodel)}")
        if i % 10 == 0:
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
            plt.plot(loss_train_axis)
            plt.plot(loss_test_axis)
            plt.show()
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

