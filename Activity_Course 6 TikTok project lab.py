#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 6 - The Nuts and bolts of machine learning**

# Recall that you are a data professional at TikTok. Your supervisor was impressed with the work you have done and has requested that you build a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.
# 
# A notebook was structured and prepared to help you in this project. A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 6 End-of-course project: Classifying videos using machine learning**
# 
# In this activity, you will practice using machine learning techniques to predict on a binary outcome variable.
# <br/>
# 
# **The purpose** of this model is to increase response time and system efficiency by automating the initial stages of the claims process.
# 
# **The goal** of this model is to predict whether a TikTok video presents a "claim" or presents an "opinion".
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Classify videos using machine learning**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 
# 1.   **What are you being asked to do? What metric should I use to evaluate success of my business/organizational objective?**
# 
# 2.   **What are the ethical implications of the model? What are the consequences of your model making errors?**
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a video does not contain a claim and it actually does)?
# 
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a video does contain a claim and it actually does not)?
# 
# 3.   **How would you proceed?**
# 

# ==> ENTER YOUR RESPONSES HERE

# **Modeling workflow and model selection process**
# 
# Previous work with this data has revealed that there are ~20,000 videos in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:
# 
# 1. Split the data into train/validation/test sets (60/20/20)
# 2. Fit models and tune hyperparameters on the training set
# 3. Perform final model selection on the validation set
# 4. Assess the champion model's performance on the test set
# 
# ![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)
# 

# ### **Task 1. Imports and data loading**
# 
# Start by importing packages needed to build machine learning models to achieve the goal of this project.

# In[2]:


# Import packages for data manipulation
import numpy as np
import pandas as pd
import pickle as pkl

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

# Import packages for data modeling
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# Now load the data from the provided csv file into a dataframe.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2: Examine data, summary info, and descriptive stats**

# Inspect the first five rows of the dataframe.

# In[4]:


# Display first few rows
data.head()


# Get the number of rows and columns in the dataset.

# In[5]:


# Get number of rows and columns
data.shape


# Get the data types of the columns.

# In[6]:


# Get data types of columns
data.dtypes


# Get basic information about the dataset.

# In[7]:


# Get basic information
data.info()


# Generate basic descriptive statistics about the dataset.

# In[8]:


# Generate basic descriptive stats
data.describe()


# Check for and handle missing values.

# In[9]:


# Check for missing values
data.isnull().sum()


# In[10]:


# Drop rows with missing values
data = data.dropna(axis=0)


# In[11]:


# Display first few rows after handling missing values
data.head()


# In[12]:


data.isnull().sum()


# Check for and handle duplicates.

# In[13]:


# Check for duplicates
data.duplicated()


# Check for and handle outliers.

# In[14]:


def outlier(col):
    
    q1 = np.percentile(data,25)
    q3 = np.percentile(dat,75)
    
    iqr = q3-q1
    lower_bound = q1 - (1.5*iqr)
    upper_bound = q3 + (1.5*iqr)
    
    outlier = [x for x in data if x<lower_bound or x>upper_bound]
    return outliers


# Tree-based models are robust to outliers, so there is no need to impute or drop any values based on where they fall in their distribution.

# 
# Check class balance.

# In[15]:


# Check class balance
data.claim_status.value_counts(normalize=True)


# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 3: Feature engineering**

# Extract the length of each `video_transcription_text` and add this as a column to the dataframe, so that it can be used as a potential feature in the model.

# In[16]:


# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
data['len_video_transcript'] = data.video_transcription_text.str.len()
data.len_video_transcript


# Calculate the average text_length for claims and opinions.

# In[17]:


# Calculate the average text_length for claims and opinions
data[['claim_status','len_video_transcript']].groupby('claim_status').mean()


# Visualize the distribution of `text_length` for claims and opinions.

# In[18]:


# Visualize the distribution of `text_length` for claims and opinions
# Create two histograms in one plot


sns.histplot(data=data,x='len_video_transcript', stat='count',kde=True, legend=True, multiple='dodge', hue='claim_status', palette='pastel')
plt.title('Video Transcript Length for Claim VS Opinion')




# **Feature selection and transformation**

# In[19]:


data.head()


# Encode target and catgorical variables.

# In[20]:


# Create a copy of the X data
df = data.copy()

# Drop unnecessary columns
df = df.drop(['#', 'video_id'], axis=1)

# Encode target variable
y = df['claim_status'].replace({'claim':1, 
                               'opinion':0})

# Dummy encode remaining categorical values
col = ['author_ban_status','verified_status']
for c in col:
    df[c] = df[c].astype('category').cat.codes


# In[21]:


df.head()


# ### **Task 4: Split the data**

# Assign target variable.

# In[22]:


# Isolate target variable
y 


# Isolate the features.

# In[23]:


# Isolate features
X = df.drop('claim_status',axis=1)

# Display first few rows of features dataframe
X.head()


# #### **Task 5: Create train/validate/test sets**

# Split data into training and testing sets, 80/20.

# In[24]:


# Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=42)


# Split the training set into training and validation sets, 75/25, to result in a final ratio of 60/20/20 for train/validate/test sets.

# In[25]:


# Split the training data into training and validation sets
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y_train, test_size=0.25, random_state=42)


# Confirm that the dimensions of the training, validation, and testing sets are in alignment.

# In[26]:


# Get shape of each training, validation, and testing set
X_tr.shape, X_test.shape, X_val.shape, y_tr.shape, y_test.shape, y_val.shape


# ### **Task 6. Build models**
# 

# ### **Build a random forest model**

# Fit a random forest model to the training set. Use cross-validation to tune the hyperparameters and select the model that performs best on recall.

# In[ ]:


# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a list of scoring metrics to capture
scoring = ['accuracy','precision','recall','f1']

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, param_grid=cv_params,scoring=scoring, cv=5, refit='recall')


# NATURAL LANGUAGE PROCIESSING NLP AHEAD
# 
# CONVERTING THE COLUMN "Video_transcription_text" INTO TOKENS

# each video's transcription text into both 2-grams and 3-grams, then takes the 10 most frequently occurring tokens from the entire dataset to use as features.

# In[28]:


from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(ngram_range=(2,3),
                           max_features=10,
                           stop_words='english')

count_vec


# Fit the vectorizer to the training data (generate the n-grams) and transform it (tally the occurrences). Only fit to the training data, not the validation or test data.

# In[29]:


count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
count_data


# Now making a seperate DataFrame with the token columns only.

# In[30]:


token_df = pd.DataFrame(data=count_data, columns = count_vec.get_feature_names_out())
token_df.head()


# Now combining the training DataFrame together with the above DataFrame

# In[31]:


#it important to drop the column 'video_transcription_text' column for which we have done the  entire tokenization process


X_train_final = pd.concat([X_train.drop(columns = ['video_transcription_text']).reset_index(drop=True), token_df], axis=1)
X_train_final.head()


# #Now this should be done same for the validation set

# But this time no fitting but just transforming .

# In[33]:


count_data_val = count_vec.transform( X_val['video_transcription_text']).toarray()
count_data_val


# In[35]:


xval_token_df = pd.DataFrame(data = count_data_val, columns = count_vec.get_feature_names_out())
xval_token_df.head()


# In[37]:


#now concatinating this DataFrame with the X_val DataFrame but dropiing the video_transcroption_text column

X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True),xval_token_df], axis=1)
X_val_final.head()


# #Now for X_test Data

# In[40]:


count_data_test = count_vec.transform(X_test['video_transcription_text']).toarray()
count_data_test


# In[42]:


test_token_df = pd.DataFrame(data = count_data_test, columns = count_vec.get_feature_names_out())
test_token_df.head()


# In[44]:


X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']).reset_index(drop=True), test_token_df], axis=1)
X_test_final.head()


# In[46]:


### Fit the model to the data 
rf_cv_model = rf_cv.fit(X_train_final,y_train)


# In[49]:


# Examine best recall score
rf_cv_model.best_score_

# Examine best parameters
rf_cv_model.best_params_
# Check the precision score to make sure the model isn't labeling everything as claims. You can do this by using the `cv_results_` attribute of the fit `GridSearchCV` object, which returns a numpy array that can be converted to a pandas dataframe. Then, examine the `mean_test_precision` column of this dataframe at the index containing the results from the best model. This index can be accessed by using the `best_index_` attribute of the fit `GridSearchCV` object.

# In[62]:


# Access the GridSearch results and convert it to a pandas df
cv_res_df = pd.DataFrame(rf_cv_model.cv_results_)

# Examine the GridSearch results df at column `mean_test_precision` in the best index
cv_res_df['mean_test_precision'][rf_cv.best_index_]


# **Question:** How well is your model performing? Consider average recall score and precision score.

# ### **Build an XGBoost model**

# In[63]:


# Instantiate the XGBoost classifier
xgb = XGBClassifier(random_state=42, objective='binary:logistic')

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a list of scoring metrics to capture
scoring = ['accuracy','f1','precision','recall']

# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, param_grid = cv_params, scoring=scoring, cv=5, refit='recall')


# In[67]:


# Fit the model to the data
xgb_cv_model = xgb_cv.fit(X_train_final,y_train)


# In[68]:


# Examine best recall score
xgb_cv_model.best_score_


# In[69]:


# Examine best parameters
xgb_cv_model.best_params_


# Repeat the steps used for random forest to examine the precision score of the best model identified in the grid search.

# In[71]:


# Access the GridSearch results and convert it to a pandas df
xgb_df = pd.DataFrame(xgb_cv_model.cv_results_)

# Examine the GridSearch results df at column `mean_test_precision` in the best index
xgb_df.mean_test_precision[xgb_cv.best_index_]


# **Question:** How well does your model perform? Consider recall score and precision score.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 7. Evaluate model**
# 
# Evaluate models against validation criteria.

# #### **Random forest**

# In[74]:


# Use the random forest "best estimator" model to get predictions on the validation set
y_val_pred = rf_cv_model.best_estimator_.predict(X_val_final)


# Display the predictions on the validation set.

# In[75]:


# Display the predictions on the validation set
y_val_pred


# Display the true labels of the validation set.

# In[77]:


# Display the true labels of the validation set
y_val.head()


# Create a confusion matrix to visualize the results of the classification model.

# In[81]:


# Create a confusion matrix to visualize the results of the classification model

# Compute values for confusion matrix
cm = confusion_matrix(y_val_pred, y_val, labels = rf_cv_model.classes_)

# Create display of confusion matrix using ConfusionMatrixDisplay()
disp = ConfusionMatrixDisplay(cm, display_labels = rf_cv_model.classes_)

# Plot confusion matrix
disp.plot()

# Display plot
plt.show()


# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the model.
# <br> </br>
# 
# **Note:** In other labs there was a custom-written function to extract the accuracy, precision, recall, and F<sub>1</sub> scores from the GridSearchCV report and display them in a table. You can also use scikit-learn's built-in [`classification_report()`](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) function to obtain a similar table of results.

# In[85]:


# Create a classification report
# Create classification report for random forest model
from sklearn.metrics import classification_report
print(classification_report(y_val, y_val_pred))


# **Question:** What does your classification report show? What does the confusion matrix indicate?
# 
# 
# The classification report above shows that the random forest model scores were perfect, but we know they weren't quite perfect. The confusion matrix indicates that there were 3 misclassifications&mdash;3 false postives and 0 false negatives.

# #### **XGBoost**
# 
# Now, evaluate the XGBoost model on the validation set.

# In[87]:


# Use the best estimator to predict on the validation data
y_pred = xgb_cv_model.best_estimator_.predict(X_val_final)
y_pred


# In[88]:


# Compute values for confusion matrix
cm = confusion_matrix(y_pred, y_val, labels = xgb_cv_model.classes_)

# Create display of confusion matrix using ConfusionMatrixDisplay()
disp = ConfusionMatrixDisplay(cm, display_labels=xgb_cv_model.classes_)

# Plot confusion matrix
disp.plot()

# Display plot
plt.show()


# In[89]:


# Create a classification report
print(classification_report(y_val, y_pred))


# **Question:** Describe your XGBoost model results. How does your XGBoost model compare to your random forest model?
# 
# The classification report above shows that the xgboost model scores were perfect, but we know they weren't quite perfect. The confusion matrix indicates that there were 17 misclassifications&mdash;16 false postives and 1 false negatives.

# ### **Use champion model to predict on test data**

# Both random forest and XGBoost model architectures resulted in nearly perfect models. Nonetheless, in this case random forest performed a little bit better, so it is the champion model.
# 

# In[92]:


y_test_pred = rf_cv_model.best_estimator_.predict(X_test_final)
y_test_pred


# In[93]:


# Compute values for confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=rf_cv_model.classes_)

# Create display of confusion matrix using ConfusionMatrixDisplay()
disp = ConfusionMatrixDisplay(cm, display_labels=rf_cv_model.classes_)

# Plot confusion matrix
disp.plot()

# Display plot
plt.show()


# #### **Feature importances of champion model**
# 

# In[97]:


importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()


# **Question:** Describe your most predictive features. Were your results surprising?
# The most predictive features all were related to engagement levels generated by the video. This is not unexpected, as analysis from prior EDA pointed to this conclusion.

# ### **Task 8. Conclusion**
# 
# In this step use the results of the models above to formulate a conclusion. Consider the following questions:
# 
# 1. **Would you recommend using this model? Why or why not?**
# 
# 2. **What was your model doing? Can you explain how it was making predictions?**
# 
# 3. **Are there new features that you can engineer that might improve model performance?**
# 
# 4. **What features would you want to have that would likely improve the performance of your model?**
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 

# Would you recommend using this model? Why or why not? Yes, one can recommend this model because it performed well on both the validation and test holdout data. Furthermore, both precision and F1 scores were consistently high. The model very successfully classified claims and opinions.
# What was your model doing? Can you explain how it was making predictions? The model's most predictive features were all related to the user engagement levels associated with each video. It was classifying videos based on how many views, likes, shares, and downloads they received.
# Are there new features that you can engineer that might improve model performance? Because the model currently performs nearly perfectly, there is no need to engineer any new features.
# What features would you want to have that would likely improve the performance of your model? The current version of the model does not need any new features. However, it would be helpful to have the number of times the video was reported. It would also be useful to have the total number of user reports for all videos posted by each author.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
