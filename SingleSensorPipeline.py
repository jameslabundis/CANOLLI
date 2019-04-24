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

NUM_EPOCHS = 40

sensors = ['hip', 'thigh', 'chest', 'thigh']

def run_kfold(X, Y, splits, model, class_labels):
    kf = KFold(n_splits=splits)
    i = 1
    acc_sum = 0
    loss_sum = 0
    cnf_tables = []
    for train_index, test_index, in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train, y_train,batch_size=128, epochs= NUM_EPOCHS,
                  verbose = False)
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
    
    return (loss_r, acc_r, cnf_tables, model)

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
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.tight_layout()
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
    # Drop not encoded data
    data = data[data[activity] != 'private/not coded']
    # Normalize dataframe
    #normalized_df = (data-data.mean())/data.std()
    # Replace na vlaues with column means
    data.fillna(data.mean(),inplace = True)
    return data
    
def run_nn(data, sensor, activity):
    model = None
    
    if activity == "broad_activity":
        model = get_model_1layer_broad()
    elif activity == "detailed_activity":
        model = get_model_1layer_detailed()
    
    x_cols = data.columns[18:(82 + 19)]
    y = data[activity]
    X = data[x_cols].values
    Y = pd.get_dummies(y).values, 50
    dummy_labels = pd.get_dummies(y).columns.tolist()
    res = run_kfold(X, Y, 10, model, dummy_labels)
    # Returns average loss, accuracy,and confusion matrix of all kfolds
    
    # Save most recent version of trained model
    last_model = res[3]
    with open(sensor + "_" + activity + "_model.json", "w") as json_file:
        json_file.write(last_model)
    # serialize weights to HDF5
    last_model.save_weights(sensor + "_" + activity +"_model.h5")
    print("Saved model to disk")
    
    return res

def conf_mat(res, sensor, activity):
    # Sum confusion matrices across all folds
    sum_cnf_matrix = np.sum(res[2], axis = 0)
    fig = plt.figure(figsize = (8,8))
    plot_confusion_matrix(sum_cnf_matrix, classes=dummy_labels,normalize = True,
                          title= activity + '-' + sensor + ' (Normalized)')
    fig.savefig("cnf_" + sensor + "_norm_" + activity  + ".jpg")
    plt.close()

    # Raw confusion matrix
    fig = plt.figure(figsize = (8,8))
    plot_confusion_matrix(sum_cnf_matrix, classes=dummy_labels,normalize = False,
                          title= activity + '-' + sensor + ' (Raw)')
    fig.savefig("cnf_" + sensor + "_raw_" + activity  + ".jpg")
    plt.close()
    
def prec_recall(res, sensor, activity):
    # Precision/Recall Table
    sum_cnf_matrix = np.sum(res[2], axis = 0)
    TP = np.diag(sum_cnf_matrix)
    FP = np.sum(sum_cnf_matrix, axis = 0) - TP
    FN = np.sum(sum_cnf_matrix, axis = 1) - TP
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    prec_recall = pd.DataFrame(data = [precision, recall],
                            columns= dummy_labels)
    prec_recall['metric'] = ['precision', 'recall']
    prec_recall.set_index('metric')
    prec_recall.to_csv("prec_recall_" + sensor + "_"+ activity + ".csv")
    
def main():
    # Run single sensor neural network for detailed activity labels
    losses = []
    accs = []
    activities = ['broad_activity', 'detailed_activity']
    for activity in activities:
        for s in sensors:
            data = clean_data(s, activity)
            res = run_nn(data, s, activity)
            sum_cnf_matrix = np.sum(res[2], axis = 0)
            conf_mat(res, s, activity)
            prec_recall(res, s, activity)

            # Save confusion matrix for all kfolds to txt file
            np.savetxt(s + "_" + activity + "_cnf_mat.txt", sum_cnf_matrix,
                       fmt = '%1.2f')
            # Append loss and accuracy for each sensor
            losses.append(res[0])
            accs.append(res[1])
        # Save sensor accuracy results to csv
        result_df = pd.DataFrame({'sensor': sensors, 
                                  'loss': losses,
                                  'accuracy': accs})
        result_df.to_csv(s + "_" + activity + "_results.csv")
    
if __name__ == '__main__':
    main()

