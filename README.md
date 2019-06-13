# CANNOLI
Scripts for data science capstone 2 project


## ActivityEncodings.ipynb
Shows the different methods for how we obtained all of the activity encodings.

get_broad_activity_type(data)
Description: creates a new column in the dataset called “broad_activity” (corresponds to broad activity in our final paper) with the broad activity encodings.

get_detailed_activity_type(data)
Description: creates a new column in the dataset called “detailed_activity” (not reflected in our final paper, does not correspond to detailed activity in our final paper) with the detailed activity encodings.

get_walking_or_running_bouts(data)
Description: creates a new column in the dataset called “walking_or_running_bouts” (used to create the next activity encoding) with the walking or running bouts. 
The column has the following values:
“<1 min walking”
“>=1min & <5min walking”
“>=5min & <10min walking”
“>=10min walking”
“<1 min running”
“>=1min & <5min running”
“>=5min & <10min running”
“>=10min running”
“” - empty if no walking or running bout was detected

get_new_encodings(data)
Description: creates a new column in the dataset called “updated_activity” (not reflected in our final paper) with the updated activity encodings.

get_final_encodings(data)
Description: creates a new column in the dataset called “final_activity” (corresponds to detailed activity in our final paper) with the final activity encodings.

get_updated_final_encodings(data)
Description: creates a new column in the dataset called “updated_final_activity” (corresponds to final activity in our final paper) with the updated final activity encodings.

TimeSeriesGraphs.ipynb
This provides function for plotting time series and coloring it by the activity type. 

Functions:

plot_all_data(data, data_label, column, column_label, folder)
Description: plots all of the data in the dataset by Mean Vector Magnitude into 5 separate plots, and colors by the column you specify.

plot_thigh_example(data, data_label, column, column_label, folder)
Description: plots a specific example in the thigh data that well demonstrates the separate updated_final_activities.

plot_thigh_example2(data, data_label, column, column_label, folder)
Description: plots a specific example in the thigh data that well demonstrates the separate updated_final_activities using the first PCA component rather than Mean Vector Magnitude.

Lastly, there is an example of how to make a simple correlation plot using the actual and predicted counts for each activity

## ConfusionMatriciesToLatex_BroadActivity.ipynb
This takes in a folder of confusion matrices that are space delimited “ “ in text files. The titling of the latex tables relies heavily on the name of the confusion matrix text file. Since we had Single Sensor and Two Sensor Results, we had two folders for confusion matrices:
SingleSensorResults
TwoSensorResults

Within each of those folders there are various folders corresponding to the activity encoding being classified:
BroadActivity
DetailedActivity
FinalActivity
UpdatedFinalActivity

Again within each of these folders I created one more folder to dump all the Latex Tables into called:
LatexTables

An example path for the text files of the confusion matrices to be converted to latex tables: 
SingleSensorResults/BroadActivity/chest_broad_activity_cnf_mat.txt

The notebook uses the “SingleSensorResults” folder to title that these are Single Sensor Results, and uses the “chest_” at the beginning of the title of the confusion matrix file to title that these are Chest Results.

The title for the above would look like: Single Sensor: Chest Broad Activity

Another example path is:
TwoSensorResults/BroadActivity/chest+wrist_broad_activity_cnf_mat.txt
The “+” is used to split the Chest and Wrist for the title. The corresponding title for this example would be: Two Sensor: Chest & Wrist Broad Activity

The ordering of the confusion matrix must be: "Bicycling", "Mixed-Activity", "Sit-Stand", "Vehicle", "Walking" with Actual on the Rows and Predicted on the Columns.


## ConfusionMatricesToLatex_DetailedActivity.ipynb
The naming schema works the same as above.

The ordering of the confusion matrix must be: "Bicycling", "Housework", "Running", "Sit/Lie","Stand and Move Light",
                      "Stand and Move Moderate or Vigorous", "Stand Still", "Vehicle",
                      "Walking Light", "Walking Moderate or Vigorous" with Actual on the Rows and Predicted on the Columns.


## ConfusionMatricesToLatex_FinalActivity.ipynb
The naming schema works the same as above.

 The ordering of the confusion matrix must be: "Sit/Lie", "Vehicle", "Stand and Move Light", 
                                              "Stand and Move Moderate or Vigorous", "Walking","Running", "Bicycling" with Actual on the Rows and Predicted on the Columns.


## ConfusionMatricesToLatex_UpdatedFinalActivity.ipynb
The naming schema works the same as above.

The ordering of the confusion matrix must be: "sit/lie", "stand and move", "walking", "running", "bicycling" with Actual on the Rows and Predicted on the Columns

## SingleSensorPipeline.py

Python script that builds simple feed forward neural networks (funnel architecure) for all four sensor types. Saves accuracy results, confusion matrices, and the neural network model itself for each sensor.

Inputs: sensor .csv files. Ex: "hip.csv"

Outputs: An overall accuracy results .csv file, a text file containing the confusion matrix, .json files/ .h5 files to store
the trained neural network

## TwoSensorPipeline.py

Python script that builds simple feed forward neural networks (funnel architecure) for two sensor combinations of the four sensors. Saves accuracy results, confusion matrices, and the neural network model itself for each sensor.

Inputs: sensor .csv files. Ex: "hip.csv"

Outputs: An overall accuracy results .csv file, a text file containing the confusion matrix, .json files/ .h5 files to store
the trained neural network

## Model_Exploration.ipynb

Jupyter notebook containing results for the random forest classifier, decision tree classifier, model Ada-Boosting, model optimization, and PCA analysis of the original 83 features.

Inputs: sensor .csv files. Ex: "hip.csv"

Outputs: Confusion matrices saved as text files for the above classifiers.


## Boosting.ipynb

Jupyter notebook containing results for Ada-Boosting the simple neural network. This notebook includes a custom class for Ada-Boosting a neural network. Also contained in this file is the grid searching done for determining the effectiveness of Ada-Boosting.

Inputs: sensor .csv files. Ex: "hip.csv"

Outputs: None.






