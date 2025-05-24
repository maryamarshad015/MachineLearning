#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a random forest model

# ## **Introduction**
# 
# 
# As you're learning, random forests are popular statistical learning algorithms. Some of their primary benefits include reducing variance, bias, and the chance of overfitting.
# 
# This activity is a continuation of the project you began modeling with decision trees for an airline. Here, you will train, tune, and evaluate a random forest model using data from spreadsheet of survey responses from 129,880 customers. It includes data points such as class, flight distance, and inflight entertainment. Your random forest model will be used to predict whether a customer will be satisfied with their flight experience.
# 
# **Note:** Because this lab uses a real dataset, this notebook first requires exploratory data analysis, data cleaning, and other manipulations to prepare it for modeling.

# ## **Step 1: Imports** 
# 

# Import relevant Python libraries and modules, including `numpy` and `pandas`libraries for data processing; the `pickle` package to save the model; and the `sklearn` library, containing:
# - The module `ensemble`, which has the function `RandomForestClassifier`
# - The module `model_selection`, which has the functions `train_test_split`, `PredefinedSplit`, and `GridSearchCV` 
# - The module `metrics`, which has the functions `f1_score`, `precision_score`, `recall_score`, and `accuracy_score`
# 

# In[89]:


# Import `numpy`, `pandas`, `pickle`, and `sklearn`.
# Import the relevant functions from `sklearn.ensemble`, `sklearn.model_selection`, and `sklearn.metrics`.

import numpy as np 
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score


# As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[90]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

### YOUR CODE HERE ###

air_data = pd.read_csv("Invistico_Airline.csv")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `read_csv()` function from the `pandas` library can be helpful here.
#  
# </details>

# Now, you're ready to begin cleaning your data. 

# ## **Step 2: Data cleaning** 

# To get a sense of the data, display the first 10 rows.

# In[91]:


# Display first 10 rows.

air_data.head(10)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `head()` function from the `pandas` library can be helpful here.
#  
# </details>

# Now, display the variable names and their data types. 

# In[92]:


# Display variable names and types.

air_data.info()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# DataFrames have an attribute that outputs variable names and data types in one result.
#  
# </details>

# **Question:** What do you observe about the differences in data types among the variables included in the data?
# 
# [Write your response here. Double-click (or enter) to edit.]

# Next, to understand the size of the dataset, identify the number of rows and the number of columns.

# In[93]:


# Identify the number of rows and the number of columns.

air_data.shape


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a method in the `pandas` library that outputs the number of rows and the number of columns in one result.
# 
# </details>

# Now, check for missing values in the rows of the data. Start with .isna() to get Booleans indicating whether each value in the data is missing. Then, use .any(axis=1) to get Booleans indicating whether there are any missing values along the columns in each row. Finally, use .sum() to get the number of rows that contain missing values.

# In[94]:


# Get Booleans to find missing values in data.
# Get Booleans to find missing values along columns.
# Get the number of rows that contain missing values.

air_data.isna().sum()


# In[95]:


air_data.isnull().any(axis=1).sum()


# **Question:** How many rows of data are missing values?**
# 
# 393 rows for the column 'Arrival Delay in Minutes' are missing or having null values.

# Drop the rows with missing values. This is an important step in data cleaning, as it makes the data more useful for analysis and regression. Then, save the resulting pandas DataFrame in a variable named `air_data_subset`.

# In[96]:


# Drop missing values.
# Save the DataFrame in variable `air_data_subset`.

air_data_subset = air_data.dropna(axis=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `dropna()` function is helpful here.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The axis parameter passed in to this function should be set to 0 (if you want to drop rows containing missing values) or 1 (if you want to drop columns containing missing values).
# </details>

# Next, display the first 10 rows to examine the data subset.

# In[97]:


# Display the first 10 rows.

air_data_subset.head(10)


# Confirm that it does not contain any missing values.

# In[98]:


# Count of missing values.

air_data_subset.isna().any(axis=1).sum()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can use the `.isna().sum()` to get the number of missing values for each variable.
# 
# </details>

# Next, convert the categorical features to indicator (one-hot encoded) features. 
# 
# **Note:** The `drop_first` argument can be kept as default (`False`) during one-hot encoding for random forest models, so it does not need to be specified. Also, the target variable, `satisfaction`, does not need to be encoded and will be extracted in a later step.

# In[99]:


# Convert categorical features to one-hot encoded features.

cols = ['Customer Type', 'Type of Travel', 'Class']

air_df = pd.get_dummies(air_data_subset, columns=cols)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can use the `pd.get_dummies()` function to convert categorical variables to one-hot encoded variables.
# </details>

# **Question:** Why is it necessary to convert categorical data into dummy variables?**
# 
# [Write your response here. Double-click (or enter) to edit.]

# Next, display the first 10 rows to review the `air_data_subset_dummies`. 

# In[100]:


# Display the first 10 rows.

air_df.head(10)


# Then, check the variables of air_data_subset_dummies.

# In[101]:


# Display variables.

air_df.dtypes


# **Question:** What changes do you observe after converting the string data to dummy variables?**
# 
# [Write your response here. Double-click (or enter) to edit.]

# ## **Step 3: Model building** 

# The first step to building your model is separating the labels (y) from the features (X).

# In[102]:


# Separate the dataset into labels (y) and features (X).

X = air_df.drop('satisfaction', axis=1)

y = air_df['satisfaction']


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Save the labels (the values in the `satisfaction` column) as `y`.
# 
# Save the features as `X`. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# To obtain the features, drop the `satisfaction` column from the DataFrame.
# 
# </details>

# Once separated, split the data into train, validate, and test sets. 

# In[103]:


# Separate into train, validate, test sets.
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#next time for splitting train data into train + validation not disturbing test data
X_tr,X_val,y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `train_test_split()` function twice to create train/validate/test sets, passing in `random_state` for reproducible results. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Split `X`, `y` to get `X_train`, `X_test`, `y_train`, `y_test`. Set the `test_size` argument to the proportion of data points you want to select for testing. 
# 
# Split `X_train`, `y_train` to get `X_tr`, `X_val`, `y_tr`, `y_val`. Set the `test_size` argument to the proportion of data points you want to select for validation. 
# 
# </details>

# ### Tune the model
# 
# Now, fit and tune a random forest model with separate validation set. Begin by determining a set of hyperparameters for tuning the model using GridSearchCV.
# 

# In[104]:


# Determine set of hyperparameters.

hyperparam = {'n_estimators' : [50,100], 
              'max_depth' : [10,50],        
              'min_samples_leaf' : [0.5,1], 
              'min_samples_split' : [0.001, 0.01],
              'max_features' : ["sqrt"], 
              'max_samples' : [.5,.9]}


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Create a dictionary `cv_params` that maps each hyperparameter name to a list of values. The GridSearch you conduct will set the hyperparameter to each possible value, as specified, and determine which value is optimal.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The main hyperparameters here include `'n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_features', and 'max_samples'`. These will be the keys in the dictionary `cv_params`.
# 
# </details>

# Next, create a list of split indices.

# In[116]:


# Create list of split indices.
from sklearn.model_selection import PredefinedSplit
split_indices = [0 if x in X_val.index else -1 for x in X_train.index]

custom_split = PredefinedSplit(split_indices)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use list comprehension, iterating over the indices of `X_train`. The list can consists of 0s to indicate data points that should be treated as validation data and -1s to indicate data points that should be treated as training data.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `PredfinedSplit()`, passing in `split_index`, saving the output as `custom_split`. This will serve as a custom split that will identify which data points from the train set should be treated as validation data during GridSearch.
# 
# </details>

# Now, instantiate your model.

# In[117]:


# Instantiate model.

rf_cv = RandomForestClassifier(random_state=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `RandomForestClassifier()`, specifying the `random_state` argument for reproducible results. This will help you instantiate a random forest model, `rf`.
# 
# </details>

# Next, use GridSearchCV to search over the specified parameters.

# In[121]:


# Search over specified parameters.

from sklearn.model_selection import GridSearchCV

rf_validation = GridSearchCV(
rf_cv,
hyperparam,
cv = custom_split,
refit = 'f1')


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `GridSearchCV()`, passing in `rf` and `cv_params` and specifying `cv` as `custom_split`. Additional arguments that you can specify include: `refit='f1', n_jobs = -1, verbose = 1`. 
# 
# </details>

# Now, fit your model.

# In[122]:


# Fit the model.

rf_validation.fit(X_train,y_train)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `fit()` method to train the GridSearchCV model on `X_train` and `y_train`. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Add the magic function `%%time` to keep track of the amount of time it takes to fit the model and display this information once execution has completed. Remember that this code must be the first line in the cell.
# 
# </details>

# Finally, obtain the optimal parameters.

# In[125]:


# Obtain optimal parameters.

rf_validation.best_params_


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `best_params_` attribute to obtain the optimal values for the hyperparameters from the GridSearchCV model.
# 
# </details>

# ## **Step 4: Results and evaluation** 

# Use the selected model to predict on your test data. Use the optimal parameters found via GridSearchCV.

# In[128]:


# Use optimal parameters on GridSearchCV.
rf_test = RandomForestClassifier(

n_estimators = 50,
max_depth = 50,
max_features = 'sqrt',
max_samples = 0.9,
min_samples_leaf = 1,
min_samples_split = 0.001,
random_state=0
)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `RandomForestClassifier()`, specifying the `random_state` argument for reproducible results and passing in the optimal hyperparameters found in the previous step. To distinguish this from the previous random forest model, consider naming this variable `rf_opt`.
# 
# </details>

# Once again, fit the optimal model.

# In[130]:


# Fit the optimal model.

rf_test.fit(X_train,y_train)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `fit()` method to train `rf_opt` on `X_train` and `y_train`.
# 
# </details>

# And predict on the test set using the optimal model.

# In[131]:


# Predict on test set.

y_pred = rf_test.predict(X_test)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `predict()` function to make predictions on `X_test` using `rf_opt`. Save the predictions now (for example, as `y_pred`), to use them later for comparing to the true labels. 
# 
# </details>

# ### Obtain performance scores

# First, get your precision score.

# In[134]:


# Get precision score.

y_test_precision = precision_score(y_test,y_pred, pos_label = 'satisfied')
y_test_precision


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `precision_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# Then, collect the recall score.

# In[135]:


# Get recall score.

y_test_recall = recall_score(y_test,y_pred, pos_label = 'satisfied')
y_test_recall


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `recall_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# Next, obtain your accuracy score.

# In[137]:


# Get accuracy score.

y_test_acc = accuracy_score(y_test,y_pred)
y_test_acc


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `accuracy_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# Finally, collect your F1-score.

# In[138]:


# Get F1 score.

y_test_f1 = f1_score(y_test,y_pred, pos_label = 'satisfied')
y_test_f1


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `f1_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# **Question:** How is the F1-score calculated?
# 
# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What are the pros and cons of performing the model selection using test data instead of a separate validation dataset?
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 

# ### Evaluate the model
# 
# Now that you have results, evaluate the model. 

# **Question:** What are the four basic parameters for evaluating the performance of a classification model?
# 
# [Write your response here. Double-click (or enter) to edit.]

# **Question:**  What do the four scores demonstrate about your model, and how do you calculate them?
# 
# [Write your response here. Double-click (or enter) to edit.]

# Calculate the scores: precision score, recall score, accuracy score, F1 score.

# In[141]:


# Precision score on test data set.

print("\nThe precision score is: {pc:.3f}".format(pc = y_test_precision), "for the test set,", "\nwhich means of all positive predictions,", "{pc_pct:.1f}% prediction are true positive.".format(pc_pct =y_test_precision * 100))


# In[142]:


# Recall score on test data set.

print("\nThe recall score is: {rc:.3f}".format(rc = y_test_recall), "for the test set,", "\nwhich means of which means of all real positive cases in test set,", "{rc_pct:.1f}% are  predicted positive.".format(rc_pct =y_test_recall * 100))


# In[143]:


# Accuracy score on test data set.

print("\nThe accuracy score is: {ac:.3f}".format(ac = y_test_acc), "for the test set,", "\nwhich means of all cases in test set,", "{ac_pct:.1f}% are predicted true positive or true negative.".format(ac_pct =y_test_acc * 100))


# In[144]:


# F1 score on test data set.

print("\nThe F1 score is: {f1:.3f}".format(f1 = y_test_f1), "for the test set,", "\nwhich means the test set's harmonic mean is {f1_pct:.1f}%.".format(f1_pct = y_test_f1 * 100))


# **Question:** How does this model perform based on the four scores?
# 
# [Write your response here. Double-click (or enter) to edit.]

# ### Evaluate the model
# 
# Finally, create a table of results that you can use to evaluate the performace of your model.

# In[150]:


# Create table of results.

results = pd.DataFrame({
    'Model':["Tuned Decision Tree", "Tuned Random Forest"],
    'F1':  [0.945422, y_test_f1],
    'Accuracy': [0.935863, y_test_recall],
    'Precision': [0.955197, y_test_precision],
    'Recall' : [0.940864, y_test_acc]
})

results


# 
# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Build a table to compare the performance of the models. Create a DataFrame using the `pd.DataFrame()` function.
# 
# </details>

# **Question:** How does the random forest model compare to the decision tree model you built in the previous lab?
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 

# ## **Considerations**
# 
# 
# **What are the key takeaways from this lab? Consider important steps when building a model, most effective approaches and tools, and overall results.**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 
# **What summary would you provide to stakeholders?**
# 
# [Write your response here. Double-click (or enter) to edit.]

# ### References

# [What is the Difference Between Test and Validation Datasets?,  Jason Brownlee](https://machinelearningmastery.com/difference-test-validation-datasets/)
# 
# [Decision Trees and Random Forests Neil Liberman](https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
