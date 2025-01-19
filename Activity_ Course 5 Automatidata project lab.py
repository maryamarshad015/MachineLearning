#!/usr/bin/env python
# coding: utf-8

# # **Automatidata project**
# **Course 5 - Regression Analysis: Simplify complex data relationships**

# The data consulting firm Automatidata has recently hired you as the newest member of their data analytics team. Their newest client, the NYC Taxi and Limousine Commission (New York City TLC), wants the Automatidata team to build a multiple linear regression model to predict taxi fares using existing data that was collected over the course of a year. The team is getting closer to completing the project, having completed an initial plan of action, initial Python coding work, EDA, and A/B testing.
# 
# The Automatidata team has reviewed the results of the A/B testing. Now it’s time to work on predicting the taxi fare amounts. You’ve impressed your Automatidata colleagues with your hard work and attention to detail. The data team believes that you are ready to build the regression model and update the client New York City TLC about your progress.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # Course 5 End-of-course project: Build a multiple linear regression model
# 
# In this activity, you will build a multiple linear regression model. As you've learned, multiple linear regression helps you estimate the linear relationship between one continuous dependent variable and two or more independent variables. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed. 
# 
# Completing this activity will help you practice planning out and buidling a multiple linear regression model based on a specific business need. The structure of this activity is designed to emulate the proposals you will likely be assigned in your career as a data professional. Completing this activity will help prepare you for those career moments.
# <br/>
# 
# **The purpose** of this project is to demostrate knowledge of EDA and a multiple linear regression model
# 
# **The goal** is to build a multiple linear regression model and evaluate the model
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** EDA & Checking Model Assumptions
# * What are some purposes of EDA before constructing a multiple linear regression model?
# 
# **Part 2:** Model Building and evaluation
# * What resources do you find yourself using as you complete this stage?
# 
# **Part 3:** Interpreting Model Results
# 
# * What key insights emerged from your model(s)?
# 
# * What business recommendations do you propose based on the models built?

# # Build a multiple linear regression model

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## PACE: **Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 

# ### Task 1. Imports and loading
# Import the packages that you've learned are needed for building linear regression models.

# In[469]:


# Imports
# Packages for numerics + dataframes
import pandas as pd
import numpy as np

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for date conversions for calculating trip durations
import datetime as dt

# Packages for OLS, MLR, confusion matrix
from statsmodels.formula.api  import ols
import sklearn.metrics as confusion_matirx
import statsmodels.api as sm


# **Note:** `Pandas` is used to load the NYC TLC dataset. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[470]:


# Load dataset into dataframe 
df0=pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv") 


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## PACE: **Analyze**
# 
# In this stage, consider the following question where applicable to complete your code response:
# 
# * What are some purposes of EDA before constructing a multiple linear regression model?
# 

# ==> ENTER YOUR RESPONSE HERE 

# ### Task 2a. Explore data with EDA
# 
# Analyze and discover data, looking for correlations, missing data, outliers, and duplicates.

# Start with `.shape` and `.info()`.

# In[471]:


# Start with `.shape` and `.info()`
df0.shape


# In[472]:


df0.info()


# Check for missing data and duplicates using `.isna()` and `.drop_duplicates()`.

# In[473]:


# Check for missing data and duplicates using .isna() and .drop_duplicates()
df0.isnull().sum()


# In[474]:


df0.drop_duplicates(keep='first', inplace=False)


# Use `.describe()`.

# In[475]:


# Use .describe()
df0.describe()


# ### Task 2b. Convert pickup & dropoff columns to datetime
# 

# In[476]:


# Check the format of the data
df0[['tpep_pickup_datetime','tpep_dropoff_datetime']].dtypes


# In[477]:


# Convert datetime columns to datetime
# Convert datetime columns to datetime
df0[['tpep_pickup_datetime','tpep_dropoff_datetime']] = df0[['tpep_pickup_datetime','tpep_dropoff_datetime']].apply(pd.to_datetime)


# ### Task 2c. Create duration column

# Create a new column called `duration` that represents the total number of minutes that each taxi ride took.

# In[478]:


# Create `duration` column
df0['duration'] = (df0['tpep_dropoff_datetime'] - df0['tpep_pickup_datetime']) / np.timedelta64(1,'m')


# ### Outliers
# 
# Call `df.info()` to inspect the columns and decide which ones to check for outliers.

# In[479]:


df0.info()


# Keeping in mind that many of the features will not be used to fit your model, the most important columns to check for outliers are likely to be:
# * `trip_distance`
# * `fare_amount`
# * `duration`
# 
# 

# ### Task 2d. Box plots
# 
# Plot a box plot for each feature: `trip_distance`, `fare_amount`, `duration`.

# In[480]:



sns.boxplot(data = df0[['trip_distance', 'fare_amount', 'duration']])


# In[481]:


fig, axes = plt.subplots(1,3 ,figsize=(15,2))
sns.boxplot(ax = axes[0], x=df0['trip_distance'])
sns.boxplot(ax = axes[1], x=df0['fare_amount'])
sns.boxplot(ax = axes[2], x=df0['duration'])
plt.show()


# **Questions:** 
# 1. Which variable(s) contains outliers? 
# 
# 2. Are the values in the `trip_distance` column unbelievable?
# 
# 3. What about the lower end? Do distances, fares, and durations of 0 (or negative values) make sense?

# 1.All three variables contain outliers.
# 
# 2.at first it might seem that values in the trip_distance column are unbelieveable but after getting knowledge that:
# It's 30 miles from the southern tip of Staten Island to the northern end of Manhattan and that's in a straight line. With this knowledge and the distribution of the values in this column, it's reasonable to leave these values alone and not alter them.But the outliers at the higher end for 'fare_amount' and 'duration' are problematic.
# 
# 3.the negative values for 'fare_amount' and 'duration' are making no sense at all.

# ### Task 2e. Imputations

# #### `trip_distance` outliers
# 
# You know from the summary statistics that there are trip distances of 0. Are these reflective of erroneous data, or are they very short trips that get rounded down?
# 
# To check, sort the column values, eliminate duplicates, and inspect the least 10 values. Are they rounded values or precise values?

# In[482]:


# Are trip distances of 0 bad data or very short trips rounded down?
df0['duration'].sort_values(ascending = True).drop_duplicates(keep='first').head(10)


# The distances are captured with a high degree of precision. However, it might be possible for trips to have distances of zero if a passenger summoned a taxi and then changed their mind. Besides, are there enough zero values in the data to pose a problem?
# 
# Calculate the count of rides where the `trip_distance` is zero.

# In[483]:


df0[df0['trip_distance'] == 0].value_counts().sum()


# #### `fare_amount` outliers

# In[484]:


df0['fare_amount'].describe()


# **Question:** What do you notice about the values in the `fare_amount` column?
# 
# Is case mein 1.5 ka factor standard formula hota hai outliers identify karne ke liye, jo zyadatar normal ya moderate variability wali datasets ke liye kaam karta hai. Lekin yahan fare_amount ka data kaafi zyada spread dikhata hai (minimum -$120 se maximum $999 tak), aur iska IQR (Interquartile Range) sirf $8 hai. Agar hum 1.5 × IQR ka use karein, toh cap $26.50 banega, jo real-world taxi fares ke liye kaafi restrictive hoga. Is wajah se yahan factor 6 use kiya gaya hai. Aap yeh samajh sakte hain:
# 
# 
# Zyada Variability ko Samajhna: Data mein fares kaafi zyada vary karte hain, jaise long rides ya surge pricing waqaira ki wajah se. Agar cap $26.50 rakha jaye toh kaafi real aur legitimate fares exclude ho jayenge.

# In[485]:


# Impute values less than $0 with 0
df0['fare_amount'] = df0['fare_amount'].apply(lambda x: 0 if x<0 else x)
df0['fare_amount'].min()


# Now impute the maximum value as `Q3 + (6 * IQR)`.

# In[486]:


def outlier_imputer(col, iqr_factor):
    
    
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    

    for col_values in col:
        # Reassign minimum to zero
        df0[col_values] = df0[col_values].apply(lambda x:0 if x<0 else x)
        # Calculate upper threshold
        
        quartile1 = df0[col_values].quantile(0.25)
        quartile3 = df0[col_values].quantile(0.75)
        iqr = quartile3 - quartile1
        threshold = quartile3 + (iqr_factor*iqr)
        
        print(f'q3:{quartile3}')
        print(f'col:{col}')
        print(f'threshold:{threshold}')
        # Reassign values > threshold to threshold
        
        df0[col_values] = df0[col_values].apply(lambda x: threshold if x>threshold else x)
        print(df0[col].describe())
        print()


# In[487]:


outlier_imputer(['fare_amount'],6)


# #### `duration` outliers
# 

# In[488]:


# Call .describe() for duration outliers
df0['duration'].describe()


# The `duration` column has problematic values at both the lower and upper extremities.
# 
# * **Low values:** There should be no values that represent negative time. Impute all negative durations with `0`.
# 
# * **High values:** Impute high values the same way you imputed the high-end outliers for fares: `Q3 + (6 * IQR)`.

# In[489]:


# Impute a 0 for any negative values
df0['duration'] = df0['duration'].apply(lambda x: 0 if x<0 else x)


# In[490]:


# Impute the high outliers
outlier_imputer(['duration'], 6)


# ### Task 3a. Feature engineering

# #### Create `mean_distance` column
# 
# When deployed, the model will not know the duration of a trip until after the trip occurs, so you cannot train a model that uses this feature. However, you can use the statistics of trips you *do* know to generalize about ones you do not know.
# 
# In this step, create a column called `mean_distance` that captures the mean distance for each group of trips that share pickup and dropoff points.
# 
# For example, if your data were:
# 
# |Trip|Start|End|Distance|
# |--: |:---:|:-:|    |
# | 1  | A   | B | 1  |
# | 2  | C   | D | 2  |
# | 3  | A   | B |1.5 |
# | 4  | D   | C | 3  |
# 
# The results should be:
# ```
# A -> B: 1.25 miles
# C -> D: 2 miles
# D -> C: 3 miles
# ```
# 
# Notice that C -> D is not the same as D -> C. All trips that share a unique pair of start and end points get grouped and averaged.
# 
# Then, a new column `mean_distance` will be added where the value at each row is the average for all trips with those pickup and dropoff locations:
# 
# |Trip|Start|End|Distance|mean_distance|
# |--: |:---:|:-:|  :--   |:--   |
# | 1  | A   | B | 1      | 1.25 |
# | 2  | C   | D | 2      | 2    |
# | 3  | A   | B |1.5     | 1.25 |
# | 4  | D   | C | 3      | 3    |
# 
# 
# Begin by creating a helper column called `pickup_dropoff`, which contains the unique combination of pickup and dropoff location IDs for each row.
# 
# One way to do this is to convert the pickup and dropoff location IDs to strings and join them, separated by a space. The space is to ensure that, for example, a trip with pickup/dropoff points of 12 & 151 gets encoded differently than a trip with points 121 & 51.
# 
# So, the new column would look like this:
# 
# |Trip|Start|End|pickup_dropoff|
# |--: |:---:|:-:|  :--         |
# | 1  | A   | B | 'A B'        |
# | 2  | C   | D | 'C D'        |
# | 3  | A   | B | 'A B'        |
# | 4  | D   | C | 'D C'        |
# 

# In[491]:


# Create `pickup_dropoff` column
df0['pickup_dropoff'] = df0['PULocationID'].astype(str).str.cat(df0['DOLocationID'].astype(str),sep='---')


# In[492]:


df0.iloc[:,15:].head()


# Now, use a `groupby()` statement to group each row by the new `pickup_dropoff` column, compute the mean, and capture the values only in the `trip_distance` column. Assign the results to a variable named `grouped`.

# In[493]:


grouped = df0.groupby('pickup_dropoff')['trip_distance'].mean()


# In[494]:


grouped


# `grouped` is an object of the `DataFrame` class.
# 
# 1. Convert it to a dictionary using the [`to_dict()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html) method. Assign the results to a variable called `grouped_dict`. This will result in a dictionary with a key of `trip_distance` whose values are another dictionary. The inner dictionary's keys are pickup/dropoff points and its values are mean distances. This is the information you want.
# 
# ```
# Example:
# grouped_dict = {'trip_distance': {'A B': 1.25, 'C D': 2, 'D C': 3}
# ```
# 
# 2. Reassign the `grouped_dict` dictionary so it contains only the inner dictionary. In other words, get rid of `trip_distance` as a key, so:
# 
# ```
# Example:
# grouped_dict = {'A B': 1.25, 'C D': 2, 'D C': 3}
#  ```

# In[495]:


# 1. Convert `grouped` to a dictionary
grouped_dict = grouped.to_dict() 
# 2. Reassign to only contain the inner dictionary
grouped_dict


# 1. Create a `mean_distance` column that is a copy of the `pickup_dropoff` helper column.
# 
# 2. Use the [`map()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html#pandas-series-map) method on the `mean_distance` series. Pass `grouped_dict` as its argument. Reassign the result back to the `mean_distance` series.
# </br></br>
# When you pass a dictionary to the `Series.map()` method, it will replace the data in the series where that data matches the dictionary's keys. The values that get imputed are the values of the dictionary.
# 
# ```
# Example:
# df['mean_distance']
# ```
# 
# |mean_distance |
# |  :-:         |
# | 'A B'        |
# | 'C D'        |
# | 'A B'        |
# | 'D C'        |
# | 'E F'        |
# 
# ```
# grouped_dict = {'A B': 1.25, 'C D': 2, 'D C': 3}
# df['mean_distance`] = df['mean_distance'].map(grouped_dict)
# df['mean_distance']
# ```
# 
# |mean_distance |
# |  :-:         |
# | 1.25         |
# | 2            |
# | 1.25         |
# | 3            |
# | NaN          |
# 
# When used this way, the `map()` `Series` method is very similar to `replace()`, however, note that `map()` will impute `NaN` for any values in the series that do not have a corresponding key in the mapping dictionary, so be careful.

# In[496]:


# 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
df0['mean_distance'] = df0['pickup_dropoff']
# 2. Map `grouped_dict` to the `mean_distance` column
df0['mean_distance'] = df0['mean_distance'].map(grouped_dict)
# Confirm that it worked
df0.iloc[:,17:].head()


# #### Create `mean_duration` column
# 
# Repeat the process used to create the `mean_distance` column to create a `mean_duration` column.

# In[497]:


grouped_duration = df0.groupby('pickup_dropoff')['duration'].mean()
# Create a dictionary where keys are unique pickup_dropoffs and values are
# mean trip duration for all trips with those pickup_dropoff combos
grouped_duration_dict_min = grouped_duration.to_dict()
grouped_duration_dict_min
# Confirm that it worked
df0['mean_duration'] = df0['pickup_dropoff']

df0['mean_duration'] = df0['mean_duration'].map(grouped_duration_dict_min)

# Confirm that it worked

df0.iloc[:,17:].head()


# In[498]:


df0['tpep_pickup_datetime'].head()


# #### Create `day` and `month` columns
# 
# Create two new columns, `day` (name of day) and `month` (name of month) by extracting the relevant information from the `tpep_pickup_datetime` column.

# In[499]:


# Create 'day' col
df0['day'] = df0['tpep_pickup_datetime'].dt.day_name()
# Create 'month' col
df0['month'] = df0['tpep_pickup_datetime'].dt.month_name()

df0['hour'] = df0['tpep_pickup_datetime'].dt.hour
df0.iloc[:,18:].head()


# #### Create `rush_hour` column
# 
# Define rush hour as:
# * Any weekday (not Saturday or Sunday) AND
# * Either from 06:00&ndash;10:00 or from 16:00&ndash;20:00
# 
# Create a binary `rush_hour` column that contains a 1 if the ride was during rush hour and a 0 if it was not.

# In[500]:


# Create 'rush_hour' col
df0['rush_hour'] = (
                ((df0['day']!='Saturday') & (df0['day']!='Sunday')) 
                                        &
    ((df0['hour']>=6) & (df0['hour']<=10)  |  (df0['hour']>=16) & df0['hour']<=20 )
).astype(int)
# If day is Saturday or Sunday, impute 0 in `rush_hour` column

df0.iloc[:, 19:].head()


# ### Task 4. Scatter plot
# 
# Create a scatterplot to visualize the relationship between `mean_duration` and `fare_amount`.

# In[501]:


# Create a scatterplot to visualize the relationship between variables of interest
sns.scatterplot(x='mean_duration', y='fare_amount', data = df0)


# The `mean_duration` variable correlates with the target variable. But what are the horizontal lines around fare amounts of 52 dollars and 63 dollars? What are the values and how many are there?
# 
# You know what one of the lines represents. 62 dollars and 50 cents is the maximum that was imputed for outliers, so all former outliers will now have fare amounts of \$62.50. What is the other line?
# 
# Check the value of the rides in the second horizontal line in the scatter plot.

# In[502]:


mask = (df0['fare_amount'] == 52)
df0[mask].value_counts().sum()


# Examine the first 30 of these trips.

# In[503]:


# Set pandas to display all columns
pd.set_option('display.max_columns', None)

df0[mask].head(30)


# **Question:** What do you notice about the first 30 trips?
# 
# 1.It seems that almost all of the trips in the first 30 rows where the fare amount was $52 either begin or end at location 132, and all of them have a RatecodeID of 2.
# 
# 2.This would seem to indicate that location 132 is in an area that frequently requires tolls to get to and from. It's likely this is an airport.
# 
# 3.The data dictionary says that RatecodeID of 2 indicates trips for JFK, which is John F. Kennedy International Airport.
# 
# 
# 4.Because `RatecodeID` is known from the data dictionary, the values for this rate code can be imputed back into the data after the model makes its predictions.

# ### Task 5. Isolate modeling variables
# 
# Drop features that are redundant, irrelevant, or that will not be available in a deployed environment.

# In[504]:


### YOUR CODE HERE ###
df0.columns


# In[505]:



df1 = df0.copy()
df1 = df1.drop(
['tpep_pickup_datetime','Unnamed: 0',
       'tpep_dropoff_datetime', 'trip_distance',
       'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
       'payment_type','extra', 'mta_tax', 'tip_amount',
       'tolls_amount', 'improvement_surcharge', 'total_amount', 'duration',
       'pickup_dropoff', 'day', 'month',
       'hour'], axis=1)

df1.info()


# ### Task 6. Pair plot
# 
# Create a pairplot to visualize pairwise relationships between `fare_amount`, `mean_duration`, and `mean_distance`.

# In[506]:


# Create a pairplot to visualize pairwise relationships between variables in the data
sns.pairplot(df1[['fare_amount', 'mean_duration', 'mean_distance']])


# These variables all show linear correlation with each other. Investigate this further.

# ### Task 7. Identify correlations

# Next, code a correlation matrix to help determine most correlated variables.

# In[507]:


# Correlation matrix to help determine most correlated variable

correlation_matrix = df1.corr()
correlation_matrix


# Visualize a correlation heatmap of the data.

# In[508]:


# Create correlation heatmap
sns.heatmap(correlation_matrix,annot = True, cmap = 'coolwarm')


# **Question:** Which variable(s) are correlated with the target variable of `fare_amount`? 
# 
# Try modeling with both variables even though they are correlated.
# 'mean_distance' and 'mean_duration' are having high correlation score of 0.91 and 0.86.
# 
# They both are also highly correlated with eachother as they have Pearson score = 0.87

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## PACE: **Construct**
# 
# After analysis and deriving variables with close relationships, it is time to begin constructing the model. Consider the questions in your PACE Strategy Document to reflect on the Construct stage.
# 

# ### Task 8a. Split data into outcome variable and features

# In[509]:


# Remove the target column from the features
X = df1.drop(columns=['fare_amount'])

# Set y variable
y = df1[['fare_amount']]

# Display first few rows
X.head()


# Set your X and y variables. X represents the features and y represents the outcome (target) variable.

# ### Task 8b. Pre-process data
# 

# Dummy encode categorical variables

# In[510]:


X.info()


# In[514]:


# Convert VendorID to string
X['VendorID'] = X['VendorID'].astype(str)
# Get dummies
# X['VendorID']=X['VendorID'].astype('category')

#or
X = pd.get_dummies(data =X ,columns=['VendorID'],drop_first=True)
X.head()


# ### Split data into training and test sets

# Create training and testing sets. The test set should contain 20% of the total samples. Set `random_state=0`.

# In[ ]:


# Create training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ### Standardize the data
# 
# Use `StandardScaler()`, `fit()`, and `transform()` to standardize the `X_train` variables. Assign the results to a variable called `X_train_scaled`.

# In[ ]:


# Standardize the X variables
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)


# ### Fit the model
# 
# Instantiate your model and fit it to the training data.

# In[515]:


# Fit your model to the training data
from sklearn.linear_model import LinearRegression

# Fit your model to the training data
lr_model=LinearRegression()
lr_model.fit(X_train_scaled, y_train)


# ### Task 8c. Evaluate model

# ### Train data
# 
# Evaluate your model performance by calculating the residual sum of squares and the explained variance score (R^2). Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[516]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Evaluate the model performance on the training data
r_sq = lr_model.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)
y_pred_train = lr_model.predict(X_train_scaled)
print('R^2:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))


# ### Test data
# 
# Calculate the same metrics on the test data. Remember to scale the `X_test` data using the scaler that was fit to the training data. Do not refit the scaler to the testing data, just transform it. Call the results `X_test_scaled`.

# In[517]:


# Scale the X_test data
X_test_scaled = scaler.transform(X_test)


# In[518]:


# Evaluate the model performance on the testing data
r_sq_test = lr_model.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
y_pred_test = lr_model.predict(X_test_scaled)
print('R^2:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))


# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## PACE: **Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### Task 9a. Results
# 
# Use the code cell below to get `actual`,`predicted`, and `residual` for the testing set, and store them as columns in a `results` dataframe.

# In[519]:


# Create a `results` dataframe
results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
results.head()


# ### Task 9b. Visualize model results

# Create a scatterplot to visualize `actual` vs. `predicted`.

# In[520]:


# Create a scatterplot to visualize `predicted` over `actual`
sns.scatterplot(y='actual', x='predicted', data=results)


# Visualize the distribution of the `residuals` using a histogram.

# In[521]:


# Visualize the distribution of the `residuals`
sns.histplot(results['residual'])


# In[522]:


# Calculate residual mean
results['residual'].mean()


# Create a scatterplot of `residuals` over `predicted`.

# In[523]:


# Create a scatterplot of `residuals` over `predicted`
sns.scatterplot(y='residual', x='predicted',data=results)


# ### Task 9c. Coefficients
# 
# Use the `coef_` attribute to get the model's coefficients. The coefficients are output in the order of the features that were used to train the model. Which feature had the greatest effect on trip fare?

# In[524]:


# Handle multiple targets
coefficients = pd.DataFrame(lr_model.coef_, columns=X.columns)
coefficients


# In[525]:


X.columns


# What do these coefficients mean? How should they be interpreted?

# Be careful here! A common misinterpretation is that for every mile traveled, the fare amount increases by a mean of $7.13. This is incorrect.
# Remember, the data used to train the model was standardized with StandardScaler().
# As such, the units are no longer miles. In other words, you cannot say "for every mile traveled...", as stated above. 
# The correct interpretation of this coefficient is: controlling for other variables, for every +1 change in standard deviation, the fare amount increases by a mean of $7.13.

# ### Task 9d. Conclusion
# 
# 1. What are the key takeaways from this notebook?
# 
# 
# 
# 2. What results can be presented from this notebook?
# 
# 

# ==> ENTER YOUR RESPONSE HERE 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 
