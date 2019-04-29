#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keras
import math
from sklearn.model_selection import KFold
import sklearn.metrics
from ann_visualizer.visualize import ann_viz
import graphviz
from sklearn.model_selection import train_test_split

sensors = ['hip', 'wrist', 'chest', 'thigh']
activities = ['broad_activity', 'detailed_activity']

def run_kfold(X, Y, splits, model, class_labels):
    kf = KFold(n_splits=splits)
    i = 1
    acc_sum = 0
    loss_sum = 0
    cnf_tables = []
    for train_index, test_index, in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train, y_train,batch_size=128, epochs=5, verbose = False)
        loss, acc = model.evaluate(X_test, y_test, steps = 1, verbose = 0)
        loss_r, acc_r = round(loss, 3), round(acc, 3)
        print("KFold: " + str(i) + ", loss: " + str(loss_r) +
         ", accuracy: " + str(acc_r))
        
        acc_sum = acc_sum + acc
        loss_sum = loss_sum + loss
        i += 1
        
        # Confusion matrix calculations
        
        # Get model predictions, save as list
        y_pred = np.argmax(model.predict(X_test, verbose = 1), axis = 1)
        y_pred = [class_labels[x] for x in y_pred.tolist()]

        # Get ground truth
        y_true = np.argmax(y_test, axis = 1)
        y_true = [class_labels[x] for x in y_true.tolist()]


        # Make confusion metrics
        cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred,
                                                      labels=class_labels)
        
        cnf_tables.append(cnf_matrix)
        
    loss_r, acc_r = round(loss_sum / splits, 3), round(acc_sum / splits, 3)
    
    return (loss_r,acc_r, cnf_tables)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.show()
    
def get_model_1layer_detailed():
    # Neural Network Architecture for single sensor
    model = tf.keras.Sequential([
            layers.Dense(83, activation="relu", input_shape = (83,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(10, activation = 'softmax')])
    # model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model_1layer_broad():
    # Neural Network Architecture for single sensor
    model = tf.keras.Sequential([
            layers.Dense(83, activation="relu", input_shape = (83,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(5, activation = 'softmax')])
    # model.summary()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def clean_data(sensor, activity):
    data = pd.read_csv(sensor + ".csv", low_memory = False)
    # Take out unnamed columns
    data.drop([col for col in data.columns if "Unnamed" in col], axis=1, inplace = True)
    # Drop not encoded activities
    data = data[data[activity] != 'private/not coded']
    # Replace na vlaues with column means
    #data.fillna(data.mean(),inplace = True)
    return data

def opt_epochs(data, sensor, activity):
    model = None
    if activity == "broad_activity":
        model = get_model_1layer_broad()
    
    elif activity == "detailed_activity":
         model = get_model_1layer_detailed()
        
    x_cols = data.columns[18:(82 + 19)]
    y = data[activity]
    X = data[x_cols]
    X = X.fillna(X.mean()).values
    Y = pd.get_dummies(y).values
    dummy_labels = pd.get_dummies(y).columns.tolist()
    
    epoch_acc(X,Y, sensor, activity, model)

def epoch_acc(X, Y, sensor, activity, model):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)

    # Optimize # of epochs
    train_acc = []
    test_acc = []
    
    for i in range(100):
        # Train model 
        hist = model.fit(X_train, y_train, batch_size=256, epochs=1, 
                         verbose = False)
        # Get train accuracy of last epoch
        train_acc.append(hist.history['acc'][len(hist.history['acc']) - 1])
        # Get test accuracy of trained model on holdout test dataset
        loss, acc = model.evaluate(X_test, y_test, steps = 1, verbose = 0)
        test_acc.append(acc)

    # Graph results
    epoch_x = np.linspace(1,100,100)
    fig = plt.figure()
    plt.plot(epoch_x, train_acc, color = "green", label = "train")
    plt.plot(epoch_x, test_acc, color = "red", label = "test")
    plt.legend()
    plt.title("Train/Test Accuracy vs Epoch: Single Sensor (" + sensor + ")")
    plt.ylim((0.75,0.95))
    plt.ylabel('Accuracy')
    plt.xlabel('# Epochs')
    fig.savefig("Train_Test_" + sensor + "_" + activity + ".jpg")
    plt.close()
    
def main():
    for activity in activities:
        for s in sensors:
            data = clean_data(s, activity)
            opt_epochs(data, s, activity)
            print(s + " completed")
    
if __name__ == '__main__':
    main()

