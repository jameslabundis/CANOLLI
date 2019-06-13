# Library dependencies
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
# Number of epochs for neural network training
NUM_EPOCHS = 30
# Sensors for training
sensors = ['hip', 'wrist', 'chest', 'thigh']

"""
Function for saving model weights and architecture.
Model weights are saved to .h5 file and architecture is saved as json.

Parameters:
sensor(string): Sensor being used for model
activity(string): Activity type being predicted
model(Keras model object): model to be saved to file

Returns:
None

"""
def save_model(sensor, activity, model):
    model_json = model.to_json()
    # Save json version of model
    with open(sensor + "_" + activity + "_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(sensor + "_" + activity +"_model.h5")
    print("Saved model to disk")

"""
Function for k-fold cross validation of model to determine accuracy.

Accuracy, loss, confusion matrices, and the model after training are stored.

Parameters:
X(ndarray): Two dimensional numpy array containing X feature data
Y(nd.array): Two dimensional numpy array containing dummified classes
sensor(string): Sensor being used for model
splits(int): Number of splits for kfold cross validation
activity(string): Activity type being predicted
class_labels(list): List containing the original class labels

Returns:
tuple(float, float, list) --> (accuracy, loss, cnf_tables)

"""
def run_kfold(X, Y, splits, sensor, activity, class_labels):
    kf = KFold(n_splits=splits)
    i = 1
    acc_sum = 0
    loss_sum = 0
    cnf_tables = []
    model = None
    for train_index, test_index, in kf.split(X):

        # Reset model and get new one
        model = get_model_1layer_updated()

        # Split data into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train, y_train,batch_size=128, epochs= NUM_EPOCHS,
                  verbose = False)

        # Evaluate accuracy of model
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


        # Make confusion matrix
        cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred,
            labels=['sit/lie', 'stand and move', 'walking', 'running',
               'bicycling'])

        # Add confusion matrix for this fold to list
        cnf_tables.append(cnf_matrix)

    # Save last trained model to disk
    save_model(sensor, activity, model)
    
    # Get average loss and accuracy for all kfolds
    loss_r, acc_r = round(loss_sum / splits, 3), round(acc_sum / splits, 3)
    
    return (loss_r, acc_r, cnf_tables)

"""
Function for making confusion matrix visual.

Parameters:

cm(float): color mapping
normalize(boolean): flag for normalizing confusion matrix
title(string): title for graph
cmap(plt object): color scheme for confusion matrix

Returns:
None

"""

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

"""
Function for creating feed forward neural network.

Parameters(None):

Returns:

keras model object

"""
def get_model_1layer_updated():
    # Neural Network Architecture for single sensor
    model = tf.keras.Sequential([
            layers.Dense(83, activation="relu", input_shape = (83,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(5, activation = 'softmax')])
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

"""
Function for initial data cleaning.

Parameters:

sensor(string): sensor name 
activity(string): activity type

Returns:

pandas Dataframe

"""

def clean_data(sensor, activity):
    data = pd.read_csv(sensor + ".csv", low_memory = False)
    # Take out unnamed columns
    data.drop([col for col in data.columns if "Unnamed" in col], axis=1, inplace = True)
    # Drop un-encoded data
    data = data[data[activity] != 'private/not coded']
    return data
    
"""
Function that runs training and validation of neural network.
Saves all validation metrics locally.

Parameters:

data(pandas DataFrame): sensor data
sensor(string): sensor type
activity(string): activity type

Returns:

tuple(float, float, list) -> (accuracy, loss, cnf_tables)
"""

def run_nn(data, sensor, activity):
    x_cols = data.columns[18:(82 + 19)]
    y = data[activity]
    # Standardize x columns
    X = data[x_cols].apply(lambda x: (x - np.mean(x)) / np.std(x))
    X = X.fillna(X.mean()).values
    # One hot vectorize categories
    Y = pd.get_dummies(y).values
    dummy_labels = pd.get_dummies(y).columns.tolist()
    # Run k-fold cross validation
    res = run_kfold(X, Y, 10, sensor, activity, dummy_labels)
    # Save confusion matrix visualizations
    #conf_mat(res, sensor, activity, dummy_labels)
    # Calculate precision/recall metrics, save results
    #prec_recall(res, sensor, activity, dummy_labels)
    return res

"""
Function that sums all confusion matrices across n folds.
Saves two versions of the confusion matrix: normalized and
raw counts.

Parameters:
res(tuple): tuple containing the accuracy, loss, and cnf_tables
sensor(string): sensor type
activity(string): activity type
dummy_labels(list): list of strings containing original class labels

Returns:
None

"""
def conf_mat(res, sensor, activity, dummy_labels):
    # Sum confusion matrices across all folds
    sum_cnf_matrix = np.sum(res[2], axis = 0)

    # Normalized confusion matrix
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
"""
Function that calculates precision and recall with given confusion matrices.
Saves precision/recall table to csv file.

Parameters:
res(tuple): tuple containing accuracy, loss, and confusion matrices
sensor(string): sensor type
activty(string): activity type
dummy_labels(list): list of strings containing original class labels

Returns:

None
"""

def prec_recall(res, sensor, activity, dummy_labels):
    # Precision/Recall Table
    sum_cnf_matrix = np.sum(res[2], axis = 0)
    # Calculate statistics
    TP = np.diag(sum_cnf_matrix)
    FP = np.sum(sum_cnf_matrix, axis = 0) - TP
    FN = np.sum(sum_cnf_matrix, axis = 1) - TP
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    prec_recall = pd.DataFrame(data = [precision, recall],
                            columns= dummy_labels)
    prec_recall['metric'] = ['precision', 'recall']
    prec_recall.set_index('metric')
    # Save precision recall table
    prec_recall.to_csv("prec_recall_" + sensor + "_"+ activity + ".csv", index = False)

"""
Function that trains and validates a neural network for each sensor type.
Saves accuracy results for all sensors to csv.

Parameters(None):

Returns:

None
"""

def main():
    # Run single sensor neural network for detailed activity labels
    activity = 'updated_final_activity'
    losses = []
    accs = []
    for s in sensors:
        data = clean_data(s, activity)
        res = run_nn(data, s, activity)
        sum_cnf_matrix = np.sum(res[2], axis = 0)

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
    result_df.to_csv(activity + "_results.csv", index = False)
    
if __name__ == '__main__':
    main()

