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
    model = NN.NNet(
        input_lay_size=X_train.shape[1],
        output_lay_size=y_train_formodel.shape[1],
        layers_size=[80, 20],
        reg_lambda=1,
        activ_func=NN.sigmoid,
        X=X_train,
        y=y_train_formodel,
        learning_rate=1,
        )
    (loss_train, loss_test, acc_train, acc_test) = NN.train(
        model, 100, X_train, y_train, X_test, y_test)
    NN.pred_val(model, 3, X_test, y_test)
