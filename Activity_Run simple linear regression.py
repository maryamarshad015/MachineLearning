#!/usr/bin/env python
# coding: utf-8

# # Activity: Run simple linear regression

# ## **Introduction**
# 
# 
# As you're learning, simple linear regression is a way to model the relationship between two variables. By assessing the direction and magnitude of a relationship, data professionals are able to uncover patterns and transform large amounts of data into valuable knowledge. This enables them to make better predictions and decisions. 
# 
# In this lab, you are part of an analytics team that provides insights about your company's sales and marketing practices. You have been assigned to a project that focuses on the use of influencer marketing. For this task, you will explore the relationship between your radio promotion budget and your sales. 
# 
# The dataset provided includes information about marketing campaigns across TV, radio, and social media, as well as how much revenue in sales was generated from these campaigns. Based on this information, company leaders will make decisions about where to focus future marketing resources. Therefore, it is critical to provide them with a clear understanding of the relationship between types of marketing campaigns and the revenue generated as a result of this investment.

# ## **Step 1: Imports** 
# 

# Import relevant Python libraries and modules.

# In[2]:


# Import relevant Python libraries and modules.

### YOUR CODE HERE ###
import pandas as pd
import seaborn as sns


# The dataset provided is a .csv file (named `marketing_sales_data.csv`), which contains information about marketing conducted in collaboration with influencers, along with corresponding sales. Assume that the numerical variables in the data are expressed in millions of dollars. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.
# 
# **Note:** This is a fictional dataset that was created for educational purposes and modified for this lab. 

# In[4]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ###

data = pd.read_csv("marketing_sales_data.csv")
data.head()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to what you learned about loading data in Python.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There is a function in the `pandas` library that allows you to read data from a .csv file and load the data into a DataFrame.
#  
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `read_csv()` function from the `pandas` library. 
# 
# </details>

# ## **Step 2: Data exploration** 
# 

# To get a sense of what the data includes, display the first 10 rows of the data.

# In[5]:


# Display the first 10 rows of the data.

### YOUR CODE HERE ###
data.head(10)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to what you learned about exploring datasets in Python.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function in the `pandas` library that allows you to get a specific number of rows from the top of a DataFrame.
#  
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `head()` function from the `pandas` library. 
# 
# </details>

# **Question:** What do you observe about the different variables included in the data?

# [Write your response here. Double-click (or enter) to edit.]

# Next, to get a sense of the size of the dataset, identify the number of rows and the number of columns.

# In[6]:


# Display number of rows, number of columns.

### YOUR CODE HERE ###
data.shape


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to what you learned about exploring datasets in Python.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There is a property in every DataFrame in `pandas` that gives you access to the number of rows and the number of columns as a tuple.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `shape` property.
# 
# </details>

# **Question:** How many rows and columns exist in the data?

# [Write your response here. Double-click (or enter) to edit.]

# Now, check for missing values in the rows of the data. This is important because missing values are not that meaningful when modeling the relationship between two variables. To do so, begin by getting Booleans that indicate whether each value in the data is missing. Then, check both columns and rows for missing values.

# In[14]:


# Start with .isna() to get booleans indicating whether each value in the data is missing.

### YOUR CODE HERE ###
data.isna()


# If you would like to read more about the `isna()` function, refer to its documentation in the references section of this lab.

# In[17]:


# Use .any(axis=1) to get booleans indicating whether there are any missing values along the columns in each row.

### YOUR CODE HERE ###
## 1 is for rows
## 0 is for columns
data.isna().any(axis=1)


# If you would like to read more about the `any()` function, refer to its documentation in the references section of this lab.

# In[25]:


# Use .sum() to get the number of rows that contain missing values.

### YOUR CODE HERE ###
data.isnull().any(axis=1).sum()


# If you would like to read more about the `sum()` function, refer to its documentation in the references section of this lab.

# **Question:** How many rows containing missing values?

# Total of 3 rows in the whole dataset are having missing values.

# Next, drop the rows that contain missing values. Data cleaning makes your data more usable for analysis and regression. Then, check to make sure that the resulting data does not contain any rows with missing values.

# In[26]:


# Use .dropna(axis=0) to indicate that you want rows which contain missing values to be dropped. To update the DataFrame, reassign it to the result.

### YOUR CODE HERE ###
data = data.dropna(axis=0)


# In[27]:


# Start with .isna() to get booleans indicating whether each value in the data is missing.
# Use .any(axis=1) to get booleans indicating whether there are any missing values along the columns in each row.
# Use .sum() to get the number of rows that contain missing values

### YOUR CODE HERE ###
data.isna().any(axis=1).sum()


# The next step for this task is checking model assumptions. To explore the relationship between radio promotion budget and sales, model the relationship using linear regression. Begin by confirming whether the model assumptions for linear regression can be made in this context. 
# 
# **Note:** Some of the assumptions can be addressed before the model is built. These will be addressed in this section. After the model is built, you will finish checking the assumptions.

# Create a plot of pairwise relationships in the data. This will help you visualize the relationships and check model assumptions. 

# In[29]:


# Create plot of pairwise relationships.

### YOUR CODE HERE ###
import seaborn as sns

sns.pairplot(data)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video section about creating a plot that shows the relationships between pairs of variables.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function in the `seaborn` library that you can call to create a plot that shows the 
#   relationships between pairs of variables.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `pairplot()` function from the `seaborn` library.
# 
# </details>

# **Question:** Is the assumption of linearity met?

# Linear Promotion between Radio Promotion and Sales exist.

# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video section about checking model assumptions for linear regression.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   Use the scatterplot of `Sales` over `Radio` found in the preceding plot of pairwise relationships. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
#   Check the scatterplot of `Sales` over `Radio` found in the plot of pairwise relationships. If the data points cluster around a line, that indicates that the assumption of linearity is met. Alternatively, if the data points resemble a random cloud or a curve, then a linear model may not fit the data.  
# 
# </details>

# ## **Step 3: Model building** 

# Select only the columns that are needed for the model.

# In[30]:


# Select relevant columns.
# Save resulting DataFrame in a separate variable to prepare for regression.

### YOUR CODE HERE ###
data_ols = data[['Radio','Sales']]


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about selecting multiple columns from a DataFrame.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   Use two pairs of square brackets around the names of the columns that should be selected.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
#   Make sure column names are spelled exactly as they are in the data.
# 
# </details>

# Now, display the first 10 rows of the new DataFrame to better understand the data.

# In[31]:


# Display first 10 rows of the new DataFrame.

### YOUR CODE HERE ###
data_ols.head(10)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about displaying contents of a DataFrame.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function in the `pandas` library that allows you to display the first n number of rows of a DataFrame, where n is a number of your choice.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
#   Call the `head()` function from the `pandas` library and pass in the number of rows from the top that you want to display. 
# 
# </details>

# Next, write the linear regression formula for modeling the relationship between the two variables of interest.

# In[36]:


# Write the linear regression formula.
# Save it in a variable.

### YOUR CODE HERE ###
#prior one y-axis and later one x-axis
ols_formula  = 'Sales ~ Radio'


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video section where model building for linear regression is discussed. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   Save the formula as string.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
#   Use a tilde to separate the y variable from the x variable so that the computer understands which is which. Make sure the spelling of each variable exactly matches the corresponding column from the data.
# 
# </details>

# Now, implement the ordinary least squares (OLS) approach for linear regression.

# In[37]:


# Implement OLS.

### YOUR CODE HERE ###
from statsmodels.formula.api import ols
OLS = ols(formula = ols_formula, data = data_ols)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video that discusses model building for linear regression.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `statsmodels` library that can be called to implement OLS.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
#   You can call the `ols()` function from the `statsmodels` library.
# 
# </details>

# Now, create a linear regression model for the data and fit the model to the data.

# In[38]:


# Fit the model to the data.
# Save the fitted model in a variable.

### YOUR CODE HERE ###
model = OLS.fit()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video section where model building for linear regression is discussed.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `statsmodels` library that can be called to fit the model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `fit()` function from the `statsmodels` library.
# 
# </details>

# ## **Step 4: Results and evaluation** 
# 

# Begin by getting a summary of the results from the model.

# In[39]:


# Get summary of results.

### YOUR CODE HERE ###
model.summary()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You may find it helpful to refer back to the video section where getting model results is discussed.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `statsmodels` library that can be called to get the summary of results from a model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `summary()` function from the `statsmodels` library.
# 
# </details>

# Next, analyze the bottom table from the results summary. Based on the table, identify the coefficients that the model determined would generate the line of best fit. The coefficients are the y-intercept and the slope. 

# **Question:** What is the y-intercept? 

# y-intercept = 41.53

# **Question:** What is the slope? 

# slope = 8.15

# **Question:** What linear equation would you write to express the relationship between sales and radio promotion budget? Use the form of y = slope * x + y-intercept? 
# 

# y = 41.53 * x + 8.15

# **Question:** What does the slope mean in this context?

# for every 1 unit increse in Radio promotion the Sales increases 8.15 million dollars.

# Now that you've built the linear regression model and fit it to the data, finish checking the model assumptions. This will help confirm your findings. First, plot the OLS data with the best fit regression line.

# In[41]:


# Plot the OLS data with the best fit regression line.

### YOUR CODE HERE ###
sns.regplot(x = 'Radio', y='Sales', data=data_ols)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about plotting data with the best fit regression line.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `seaborn` library that can be useful here.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `regplot()` function from the `seaborn` library.
# 
# </details>

# **Question:** What do you observe from the preceding regression plot?

# The two varibales confirms the asumption of linearity.

# Now, check the normality assumption. Get the residuals from the model.

# In[43]:


# Get the residuals from the model.

### YOUR CODE HERE ###
residuals = model.resid


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about accessing residuals.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is an attribute from the `statsmodels` library that can be called to get the residuals from a fitted model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `resid` attribute from the `statsmodels` library.
# 
# </details>

# Now, visualize the distribution of the residuals.

# In[51]:


# Visualize the distribution of the residuals.

### YOUR CODE HERE ##
#histogram plot of residuals
import matplotlib.pyplot as plt
fig = sns.histplot(residuals)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Residuals Distribution')
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about visualizing residuals.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `seaborn` library that can be called to create a histogram.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `histplot()` function from the `seaborn` library.
# 
# </details>

# **Question:** Based on the visualization, what do you observe about the distribution of the residuals?

# Satisfies the Normality Assumption.

# Next, create a Q-Q plot to confirm the assumption of normality.

# In[53]:


# Create a Q-Q plot.

### YOUR CODE HERE ###
import statsmodels.api as sm
sm.qqplot(residuals, line='s')
plt.title('QQ-Plot')
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about creating a Q-Q plot.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `statsmodels` library that can be called to create a Q-Q plot.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `qqplot()` function from the `statsmodels` library.
# 
# </details>

# **Question:** Is the assumption of normality met?

# The assumption of Normaltiy is met because of straight diagonal line of residuals.

# Now, check the assumptions of independent observation and homoscedasticity. Start by getting the fitted values from the model.

# In[56]:


# Get fitted values.

### YOUR CODE HERE ###
X = data_ols[['Radio']]
fitted_values = model.predict(X)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about calculating fitted values.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `statsmodels` library that can be called to calculate fitted values from the model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `predict()` function from the `statsmodels` library. Make sure to pass in the column from `ols_data` corresponding to the x variable.
# 
# </details>

# Next, create a scatterplot of the residuals against the fitted values.

# In[63]:


# Create a scatterplot of residuals against fitted values.

### YOUR CODE HERE ###
sns.scatterplot(x = fitted_values, y= residuals)
plt.axhline(0,color = 'red')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals VS Fitted Values')
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the video about visualizing residuals against fitted values.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
#   There is a function from the `seaborn` library that can be called to create a scatterplot.
# 
# </details>

# <details>
#   <summary><h4>Hint 3</h4></summary>
# 
# Call the `scatterplot()` function from the `seaborn` library.
# 
# </details>

# **Question:** Are the assumptions of independent observation and homoscedasticity met?
# 

# In the preceding scatterplot, the data points have a cloud-like resemblance and do not follow an explicit pattern. So it appears that the independent observation assumption has not been violated. Given that the residuals appear to be randomly spaced, the homoscedasticity assumption seems to be met.

# ## **Considerations**

# **What are some key takeaways that you learned during this lab?**

# [Write your response here. Double-click (or enter) to edit.]

# **How would you present your findings from this lab to others?**

# [Write your response here. Double-click (or enter) to edit.]

# **What summary would you provide to stakeholders?**

# [Write your response here. Double-click (or enter) to edit.]

# **References**
# 
# [Pandas.DataFrame.Any — Pandas 1.4.3 Documentation.](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.any.html)
# 
# [Pandas.DataFrame.Isna — Pandas 1.4.3 Documentation.](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html)
# 
# [Pandas.Series.Sum — Pandas 1.4.3 Documentation.](https://pandas.pydata.org/docs/reference/api/pandas.Series.sum.html)
# 
# [Saragih, H.S. *Dummy Marketing and Sales Data*.](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
