#!/usr/bin/env python
# coding: utf-8

# # Activity: Perform multiple linear regression
# 

# ## Introduction

# As you have learned, multiple linear regression helps you estimate the linear relationship between one continuous dependent variable and two or more independent variables. For data science professionals, this is a useful skill because it allows you to compare more than one variable to the variable you're measuring against. This provides the opportunity for much more thorough and flexible analysis. 
# 
# For this activity, you will be analyzing a small business' historical marketing promotion data. Each row corresponds to an independent marketing promotion where their business uses TV, social media, radio, and influencer promotions to increase sales. They previously had you work on finding a single variable that predicts sales, and now they are hoping to expand this analysis to include other variables that can help them target their marketing efforts.
# 
# To address the business' request, you will conduct a multiple linear regression analysis to estimate sales from a combination of independent variables. This will include:
# 
# * Exploring and cleaning data
# * Using plots and descriptive statistics to select the independent variables
# * Creating a fitting multiple linear regression model
# * Checking model assumptions
# * Interpreting model outputs and communicating the results to non-technical stakeholders

# ## Step 1: Imports

# ### Import packages

# Import relevant Python libraries and modules.

# In[31]:


# Import libraries and modules.

### YOUR CODE HERE ### 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm


# ### Load dataset

# `Pandas` was used to load the dataset `marketing_sales_data.csv` as `data`, now display the first five rows. The variables in the dataset have been adjusted to suit the objectives of this lab. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[32]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ### 
data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.

### YOUR CODE HERE ### 

data.head(5)


# ## Step 2: Data exploration

# ### Familiarize yourself with the data's features
# 
# Start with an exploratory data analysis to familiarize yourself with the data and prepare it for modeling.
# 
# The features in the data are:
# 
# * TV promotional budget (in "Low," "Medium," and "High" categories)
# * Social media promotional budget (in millions of dollars)
# * Radio promotional budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in "Mega," "Macro," "Micro," and "Nano" categories)
# 

# **Question:** What are some purposes of EDA before constructing a multiple linear regression model?

# [Write your response here. Double-click (or enter) to edit.]

# ### Create a pairplot of the data
# 
# Create a pairplot to visualize the relationship between the continous variables in `data`.

# In[33]:


# Create a pairplot of the data.

### YOUR CODE HERE ### 
sns.pairplot(data)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content where creating a pairplot is demonstrated](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/item/dnjWm).
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a pairplot showing the relationships between variables in the data.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `pairplot()` function from the `seaborn` library and pass in the entire DataFrame.
# 
# </details>
# 

# **Question:** Which variables have a linear relationship with `Sales`? Why are some variables in the data excluded from the preceding plot?
# 
# 

# Radio has linear relationship with the Sales and TV  and influencer variables are removed from the preceding plot because they are non numeric and not categorical or continuous.

# ### Calculate the mean sales for each categorical variable

# There are two categorical variables: `TV` and `Influencer`. To characterize the relationship between the categorical variables and `Sales`, find the mean `Sales` for each category in `TV` and the mean `Sales` for each category in `Influencer`. 

# In[37]:


# Calculate the mean sales for each TV category. 

### YOUR CODE HERE ### 

avg_tv_sales = data.groupby('TV')['Sales'].mean()

# Calculate the mean sales for each Influencer category. 

### YOUR CODE HERE ### 
avg_influencer_sales = data.groupby('Influencer')['Sales'].mean()

print(avg_tv_sales)
print('\n')
print(avg_influencer_sales)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Find the mean `Sales` when the `TV` promotion is `High`, `Medium`, or `Low`.
#     
# Find the mean `Sales` when the `Influencer` promotion is `Macro`, `Mega`, `Micro`, or `Nano`.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `groupby` operation in `pandas` to split an object (e.g., data) into groups and apply a calculation to each group.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# To calculate the mean `Sales` for each `TV` category, group by `TV`, select the `Sales` column, and then calculate the mean. 
#     
# Apply the same process to calculate the mean `Sales` for each `Influencer` category.
# 
# </details>

# **Question:** What do you notice about the categorical variables? Could they be useful predictors of `Sales`?
# 
# 

# The average Sales for High TV promotions is considerably higher than for Medium and Low TV promotions. TV may be a strong predictor of Sales.
# 
# The categories for Influencer have different average Sales, but the variation is not substantial. Influencer may be a weak predictor of Sales.
# 
# These results can be investigated further when fitting the multiple linear regression model.

# ### Remove missing data
# 
# This dataset contains rows with missing values. To correct this, drop all rows that contain missing data.

# In[38]:


# Drop rows that contain missing data and update the DataFrame.

### YOUR CODE HERE ### 

data = data.dropna(axis = 0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `pandas` function that removes missing values.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `dropna()` function removes missing values from an object (e.g., DataFrame).
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `data.dropna(axis=0)` to drop all rows with missing values in `data`. Be sure to properly update the DataFrame.
# 
# </details>

# ### Clean column names

# The `ols()` function doesn't run when variable names contain a space. Check that the column names in `data` do not contain spaces and fix them, if needed.

# In[42]:


# Rename all columns in data that contain a space. 

### YOUR CODE HERE ### 
data.columns
data = data.rename(columns={'Social Media':'Social_Media'})


# In[43]:


data.columns


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is one column name that contains a space. Search for it in `data`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `Social Media` column name in `data` contains a space. This is not allowed in the `ols()` function.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `rename()` function in `pandas` and use the `columns` argument to provide a new name for `Social Media`.
# 
# </details>

# ## Step 3: Model building

# ### Fit a multiple linear regression model that predicts sales
# 
# Using the independent variables of your choice, fit a multiple linear regression model that predicts `Sales` using two or more independent variables from `data`.

# In[44]:


# Define the OLS formula.

ols_formula = 'Sales ~ Radio + C(TV)'

# Create an OLS model.

OLS = ols(formula = ols_formula, data = data)

# Fit the model.

model = OLS.fit()

# Save the results summary.

results = model.summary()

# Display the model results.

results


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the content that discusses [model building](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/lecture/zd74V/interpret-multiple-regression-coefficients) for linear regression.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `ols()` function imported earlier—which creates a model from a formula and DataFrame—to create an OLS model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# You previously learned how to specify in `ols()` that a feature is categorical. 
#     
# Be sure the string names for the independent variables match the column names in `data` exactly.
# 
# </details>

# **Question:** Which independent variables did you choose for the model, and why?
# 
# 

# TV was selected, as the preceding analysis showed a strong relationship between the TV promotional budget and the average Sales.
# 
# Radio was selected because the pairplot showed a strong linear relationship between Radio and Sales.
# 
# Social Media was not selected because it did not increase model performance and it was later determined to be correlated with another independent variable: Radio.
# 
# Influencer was not selected because it did not show a strong relationship to Sales in the preceding analysis

# ### Check model assumptions

# For multiple linear regression, there is an additional assumption added to the four simple linear regression assumptions: **multicollinearity**. 
# 
# Check that all five multiple linear regression assumptions are upheld for your model.

# ### Model assumption: Linearity

# Create scatterplots comparing the continuous independent variable(s) you selected previously with `Sales` to check the linearity assumption. Use the pairplot you created earlier to verify the linearity assumption or create new scatterplots comparing the variables of interest.

# In[64]:


# Create a scatterplot for each independent variable and the dependent variable.

### YOUR CODE HERE ### 
fig, axes = plt.subplots(1,4, figsize=(10,4))


sns.scatterplot(x=data['Radio'], y=data['Sales'],ax = axes[0])
axes[0].set_title('Radio VS Sales')

sns.scatterplot(x=data['TV'], y=data['Sales'],ax=axes[1])
axes[1].set_title('TV VS Sales')


sns.scatterplot(x=data['Influencer'], y=data['Sales'],ax=axes[2])
axes[2].set_title('Influencer VS Sales')


sns.scatterplot(x=data['Social_Media'], y=data['Sales'],ax=axes[3])
axes[3].set_title('Social_Media VS Sales')




# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a scatterplot to display the values for two variables.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `scatterplot()` function in `seaborn`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
#     
# Pass the independent and dependent variables in your model as the arguments for `x` and `y`, respectively, in the `scatterplot()` function. Do this for each continous independent variable in your model.
# 
# </details>

# **Question:** Is the linearity assumption met?
# 

# The Linearity assumption between Radio and Sales are met.Social  Media is also seem to have linear relationship with Sales but becuase of the fact that it has collinearity withb other independent variable Sales hence it is not included in the features.

# ### Model assumption: Independence

# The **independent observation assumption** states that each observation in the dataset is independent. As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# ### Model assumption: Normality

# Create the following plots to check the **normality assumption**:
# 
# * **Plot 1**: Histogram of the residuals
# * **Plot 2**: Q-Q plot of the residuals

# In[71]:


# Calculate the residuals.

residuals = model.resid

# Create a histogram with the residuals. 

sns.histplot(residuals)
plt.xlabel('Resiuals')
plt.title('Histogram of Resiuals')
# Create a Q-Q plot of the residuals.

sm.qqplot(residuals,line='s')
plt.title('Normality Q-Q Plot')


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the residuals from the fit model object.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.resid` to get the residuals from a fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# For the histogram, pass the residuals as the first argument in the `seaborn` `histplot()` function.
#     
# For the Q-Q plot, pass the residuals as the first argument in the `statsmodels` `qqplot()` function.
# 
# </details>

# **Question:** Is the normality assumption met?
# 
# 

# The normality assumption met as histogram shows normal distributin of residuals and the qq plot shows almost diagonal line.

# ### Model assumption: Constant variance

# Check that the **constant variance assumption** is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.

# In[81]:


# Create a scatterplot with the fitted values from the model and the residuals.

sns.scatterplot(x=model.fittedvalues, y=residuals)

plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

plt.title('Homoscedasticity Assumption Check')
plt.axhline(0,color = 'red')

# Add a line at y = 0 to visualize the variance of residuals above and below 0.


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the fitted values from the model object fit earlier.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.fittedvalues` to get the fitted values from a fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# 
# Call the `scatterplot()` function from the `seaborn` library and pass in the fitted values and residuals.
#     
# Add a line to a figure using the `axline()` function.
# 
# </details>

# **Question:** Is the constant variance assumption met?
# 
# 
# 

# 
# The fitted values are in three cloud groups namely categories : {'low','medium','high'} because the categorical variable is dominating in this model, meaning that TV is the biggest factor that decides the sales.
# 
# However, the variance where there are fitted values is similarly distributed, validating that the assumption is met.
# 

# ### Model assumption: No multicollinearity

# The **no multicollinearity assumption** states that no two independent variables ($X_i$ and $X_j$) can be highly correlated with each other. 
# 
# Two common ways to check for multicollinearity are to:
# 
# * Create scatterplots to show the relationship between pairs of independent variables
# * Use the variance inflation factor to detect multicollinearity
# 
# Use one of these two methods to check your model's no multicollinearity assumption.

# In[82]:


# Create a pairplot of the data.

sns.pairplot(data)


# In[84]:


# Calculate the variance inflation factor (optional).

from statsmodels.stats.outliers_influence import variance_inflation_factor

X=data[['Social_Media','Radio']]

# a list containg vif calcukated for each and every row of Variables
vif = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]


#creating a dataframe for the vif of the two variables
vif_df = pd.DataFrame(vif, index = X.columns, columns = ['VIF'])
vif_df


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Confirm that you previously created plots that could check the no multicollinearity assumption.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `pairplot()` function applied earlier to `data` plots the relationship between all continous variables  (e.g., between `Radio` and `Social Media`).
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# The `statsmodels` library has a function to calculate the variance inflation factor called `variance_inflation_factor()`. 
#     
# When using this function, subset the data to only include the continous independent variables (e.g., `Radio` and `Social Media`). Refer to external tutorials on how to apply the variance inflation factor function mentioned previously.
#  
# 
# </details>

# **Question 8:** Is the no multicollinearity assumption met?
# 
# The scatterplot between Social Media and Radio shows slight multicollinearity becuase of the slight linear relationship visible in the scatterplot and violates the no multi-collinearity assumption because of the RIF value = 5.17 which is slightly high.
# 

# ## Step 4: Results and evaluation

# ### Display the OLS regression results
# 
# If the model assumptions are met, you can interpret the model results accurately.
# 
# First, display the OLS regression results.

# In[85]:


# Display the model results summary.

results


# **Question:** What is your interpretation of the model's R-squared?
# 

# R squared value is 90% which means that independent variables explain 90% of the variance in the dependent variable Y (Sales).

# ### Interpret model coefficients

# With the model fit evaluated, you can look at the coefficient estimates and the uncertainty of these estimates.
# 
# Again, display the OLS regression results.

# In[86]:


# Display the model results summary.

results


# **Question:** What are the model coefficients?
# 
# 

# B0 = 218.52  , B1(LOW)=-154.29 , B2(MEDIUM) = -75.3, B(RADIO) = 2.9

# **Question:** How would you write the relationship between `Sales` and the independent variables as a linear equation?
# 
# 

# Y(SALES) = 218.52 +2.9*RADIO -154.29*TV(LOW)-75.3*TV(MEDIUM)

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?
# 
# 

# The coeffecients are statistically significant because p-values are 0.000 which is less than 0.05.

# **Question:** Why is it important to interpret the beta coefficients?
# 
# 

# In order to predict the best-fit line.

# **Question:** What are you interested in exploring based on your model?
# 
# 

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** Do you think your model could be improved? Why or why not? How?

# [Write your response here. Double-click (or enter) to edit.]

# ## Conclusion

# **What are the key takeaways from this lab?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# **What results can be presented from this lab?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# **How would you frame your findings to external stakeholders?**
# 
# High TV promotional budgets have a substantial positive influence on sales. The model estimates that switching from a high to medium TV promotional budget reduces sales by $\$75.3120$ million (95% CI $[-82.431,-68.193])$, and switching from a high to low TV promotional budget reduces sales by $\$154.297$ million (95% CI $[-163.979,-144.616])$. The model also estimates that an increase of $\$1$ million in the radio promotional budget will yield a $\$2.9669$ million increase in sales (95% CI $[2.551,3.383]$).
# 
# Thus, it is recommended that the business allot a high promotional budget to TV when possible and invest in radio promotions to increase sales. 
# 

# #### **References**
# 
# Saragih, H.S. (2020). [*Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data).

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
