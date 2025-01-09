#!/usr/bin/env python
# coding: utf-8

# # Activity: Hypothesis testing with Python

# ## **Introduction**
# 

# As you've been learning, analysis of variance (commonly called ANOVA) is a group of statistical techniques that test the difference of means among three or more groups. It's a powerful tool for determining whether population means are different across groups and for answering a wide range of business questions.
# 
# In this activity, you are a data professional working with historical marketing promotion data. You will use the data to run a one-way ANOVA and a post hoc ANOVA test. Then, you will communicate your results to stakeholders. These experiences will help you make more confident recommendations in a professional setting. 
# 
# In your dataset, each row corresponds to an independent marketing promotion, where your business uses TV, social media, radio, and influencer promotions to increase sales. You have previously provided insights about how different promotion types affect sales; now stakeholders want to know if sales are significantly different among various TV and influencer promotion types.
# 
# To address this request, a one-way ANOVA test will enable you to determine if there is a statistically significant difference in sales among groups. This includes:
# * Using plots and descriptive statistics to select a categorical independent variable
# * Creating and fitting a linear regression model with the selected categorical independent variable
# * Checking model assumptions
# * Performing and interpreting a one-way ANOVA test
# * Comparing pairs of groups using an ANOVA post hoc test
# * Interpreting model outputs and communicating the results to nontechnical stakeholders

# ## **Step 1: Imports** 
# 

# Import pandas, pyplot from matplotlib, seaborn, api from statsmodels, ols from statsmodels.formula.api, and pairwise_tukeyhsd from statsmodels.stats.multicomp.

# In[1]:


# Import libraries and packages.

### YOUR CODE HERE ### 
import pandas as pd
import seaborn as sns


# `Pandas` was used to load the dataset `marketing_sales_data.csv` as `data`, now display the first five rows. The variables in the dataset have been adjusted to suit the objectives of this lab. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ### 
data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.

data.head()


# The features in the data are:
# * TV promotion budget (in Low, Medium, and High categories)
# * Social media promotion budget (in millions of dollars)
# * Radio promotion budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in Mega, Macro, Nano, and Micro categories)

# **Question:** Why is it useful to perform exploratory data analysis before constructing a linear regression model?

# [Write your response here. Double-click (or enter) to edit.]

# ## **Step 2: Data exploration** 
# 

# First, use a boxplot to determine how `Sales` vary based on the `TV` promotion budget category.

# In[3]:


# Create a boxplot with TV and Sales.

sns.boxplot(x='TV',y='Sales', data=data)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `seaborn` library that creates a boxplot showing the distribution of a variable across multiple groups.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `boxplot()` function from `seaborn`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `TV` as the `x` argument, `Sales` as the `y` argument, and `data` as the `data` argument.
# 
# </details>

# **Question:** Is there variation in `Sales` based off the `TV` promotion budget?

# There is a clear vsriation in TV promotion budget with Sales but we dont know wether the variation is statistically significant.

# Now, use a boxplot to determine how `Sales` vary based on the `Influencer` size category.

# In[4]:


# Create a boxplot with Influencer and Sales.

sns.boxplot(x='Influencer',y='Sales', data=data)


# **Question:** Is there variation in `Sales` based off the `Influencer` size?

# No Visible variation but we dont know even if a variation exists wether it would be statistically significant or not?

# ### Remove missing data
# 
# You may recall from prior labs that this dataset contains rows with missing values. To correct this, drop these rows. Then, confirm the data contains no missing values.

# In[5]:


# Drop rows that contain missing data and update the DataFrame.

data.dropna(inplace=True)

# Confirm the data contains no missing values.

data.isna().sum()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a `pandas` function that removes missing values.
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
# Verify the data is updated properly after the rows containing missing data are dropped.
# 
# </details>

# ## **Step 3: Model building** 
# 

# Fit a linear regression model that predicts `Sales` using one of the independent categorical variables in `data`. Refer to your previous code for defining and fitting a linear regression model.

# In[6]:


from statsmodels.formula.api import ols
import statsmodels.api as sm


# In[13]:


# Define the OLS formula.

ols_formula = 'Sales ~ C(TV)'

# Create an OLS model.

OLS = ols(formula = ols_formula, data=data)

# Fit the model.
model = OLS.fit()
# Save the results summary.

results = model.summary()

# Display the model results.

results


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to code you've written to fit linear regression models.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `ols()` function from `statsmodels.formula.api`, which creates a model from a formula and DataFrame, to create an OLS model.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `C()` around the variable name in the ols formula to indicate a variable is categorical.
#     
# Be sure the variable string names exactly match the column names in `data`.
# 
# </details>

# **Question:** Which categorical variable did you choose for the model? Why?

# 
# * `TV` was selected as the preceding analysis showed a strong relationship between the `TV` promotion budget and the average `Sales`.
# * `Influencer` was not selected because it did not show a strong relationship to `Sales` in the analysis.

# ### Check model assumptions

# Now, check the four linear regression assumptions are upheld for your model.

# **Question:** Is the linearity assumption met?

# 
# Because your model does not have any continuous independent variables, the linearity assumption is not required. 

# The independent observation assumption states that each observation in the dataset is independent. As each marketing promotion (row) is independent from one another, the independence assumption is not violated.

# Next, verify that the normality assumption is upheld for the model.

# In[19]:


# Calculate the residuals.
import matplotlib.pyplot as plt

residuals = model.resid

# Create a histogram with the residuals. 
fig,axes = plt.subplots(1,2,figsize=(6,3))
sns.histplot(residuals,ax=axes[0])

# Create a QQ plot of the residuals.

sm.qqplot(residuals,line='s',ax=axes[1])
plt.tight_layout()
plt.show()


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
# For the QQ-plot, pass the residuals as the first argument in the `statsmodels` `qqplot()` function.
# 
# </details>

# **Question:** Is the normality assumption met?

# The normality assumption met for sure as there is a diagonal line for residuals in the qqplot and histogram shows the normal distribution.

# Now, verify the constant variance (homoscedasticity) assumption is met for this model.

# In[24]:


# Create a scatter plot with the fitted values from the model and the residuals.

sns.scatterplot(x=model.fittedvalues, y=residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
# Add a line at y = 0 to visualize the variance of residuals above and below 0.

plt.axhline(0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the fitted values from the model object fit earlier.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.fittedvalues` to get the fitted values from the fit model called `model`.
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

# **Question:** Is the constant variance (homoscedasticity) assumption met?

# The variance is constant among fitted values hence ansuring homoscedasticity assumption met successfully.

# ## **Step 4: Results and evaluation** 

# First, display the OLS regression results.

# In[25]:


# Display the model results summary.

results


# **Question:** What is your interpretation of the model's R-squared?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** Do you think your model could be improved? Why or why not? How?

# [Write your response here. Double-click (or enter) to edit.]

# ### Perform a one-way ANOVA test
# 
# With the model fit, run a one-way ANOVA test to determine whether there is a statistically significant difference in `Sales` among groups. 

# In[37]:


# Create an one-way ANOVA table for the fit model.
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

sm.stats.anova_lm(model, typ=2)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a one-way ANOVA test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There is a function in `statsmodels.api` (i.e. `sm`) that peforms an ANOVA test for a fit linear model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `anova_lm()` function from `sm.stats`. Specify the type of ANOVA test (for example, one-way or two-way), using the `typ` parameter.
#    
# 
# </details>

# **Question:** What are the null and alternative hypotheses for the ANOVA test?

# N(o) Null = The Sales is same and equal for all the TV Categories
# 
# N(1) Alternative = The Sales is not same for all the TV Categories

# **Question:** What is your conclusion from the one-way ANOVA test?

# As the p-Value is too small hence rejecting the null hypothesis and declaring that there is a difference in Sales for different TV Categories.

# **Question:** What did the ANOVA test tell you?

# 
# The results of the one-way ANOVA test indicate that you can reject the null hypothesis in favor of the alternative hypothesis. There is a statistically significant difference in `Sales` among `TV` groups.

# ### Perform an ANOVA post hoc test
# 
# If you have significant results from the one-way ANOVA test, you can apply ANOVA post hoc tests such as the Tukey’s HSD post hoc test. 
# 
# Run the Tukey’s HSD post hoc test to compare if there is a significant difference between each pair of categories for TV.

# In[42]:


# Perform the Tukey's HSD post hoc test.

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_hsd_posthoc = pairwise_tukeyhsd(endog = data['Sales'], groups = data['TV'])

tukey_hsd_posthoc.summary()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a Tukey's HSD post hoc test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `pairwise_tukeyhsd()` function from `statsmodels.stats.multicomp`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# The `endog` argument in `pairwise_tukeyhsd` indicates which variable is being compared across groups (i.e., `Sales`). The `groups` argument in `pairwise_tukeyhsd` tells the function which variable holds the group you’re interested in reviewing.
# 
# </details>

# **Question:** What is your interpretation of the Tukey HSD test?

# Rejecting the null hypo for High and Low TV budget promotion categories meaning there is differnce in Sales between Low and High TV Categories and similar for the other two groups meaning there is differnce in Sales for these two groups as well.

# **Question:** What did the post hoc tell you?**

# A post hoc test was conducted to determine which `TV` groups are different and how many are different from each other. This provides more detail than the one-way ANOVA results, which can at most determine that at least one group is different. Further, using the Tukey HSD controls for the increasing probability of incorrectly rejecting a null hypothesis from peforming multiple tests. 
# 
# The results were that `Sales` is not the same between any pair of `TV` groups. 

# ## **Considerations**
# 
# **What are some key takeaways that you learned during this lab?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 
# **What summary would you provide to stakeholders? Consider the statistical significance of key relationships and differences in distribution.**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 

# #### **Reference**
# [Saragih, H.S. *Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
