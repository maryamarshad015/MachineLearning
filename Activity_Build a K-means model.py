#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a K-means model 
# 
# ## **Introduction**
# 
# K-means clustering is very effective when segmenting data and attempting to find patterns. Because clustering is used in a broad array of industries, becoming proficient in this process will help you expand your skillset in a widely applicable way.   
# 
# In this activity, you are a consultant for a scientific organization that works to support and sustain penguin colonies. You are tasked with helping other staff members learn more about penguins in order to achieve this mission. 
# 
# The data for this activity is in a spreadsheet that includes datapoints across a sample size of 345 penguins, such as species, island, and sex. Your will use a K-means clustering model to group this data and identify patterns that provide important insights about penguins.
# 
# **Note:** Because this lab uses a real dataset, this notebook will first require basic EDA, data cleaning, and other manipulations to prepare the data for modeling. 

# ## **Step 1: Imports** 
# 

# Import statements including `K-means`, `silhouette_score`, and `StandardScaler`.

# In[67]:


# Import standard operational packages.
import numpy as np 
import pandas as pd

# Important tools for modeling and evaluation.
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Import visualization packages.

import seaborn as sns


# `Pandas` is used to load the penguins dataset, which is built into the `seaborn` library. The resulting `pandas` DataFrame is saved in a variable named `penguins`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[68]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

# Save the `pandas` DataFrame in variable `penguins`. 

### YOUR CODE HERE ###

penguins = pd.read_csv("penguins.csv")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `load_dataset` function. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The function is from seaborn (`sns`). It should be passed in the dataset name `'penguins'` as a string. 
# 
# </details>

# Now, review the first 10 rows of data.
# 

# In[69]:


# Review the first 10 rows.

penguins.head(10)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `head()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# By default, the method only returns five rows. To change this, specify how many rows `(n = )` you want.
# 
# </details>

# ## **Step 2: Data exploration** 
# 
# After loading the dataset, the next step is to prepare the data to be suitable for clustering. This includes: 
# 
# *   Exploring data
# *   Checking for missing values
# *   Encoding data 
# *   Dropping a column
# *   Scaling the features using `StandardScaler`

# ### Explore data
# 
# To cluster penguins of multiple different species, determine how many different types of penguin species are in the dataset.

# In[70]:


# Find out how many penguin types there are.

penguins['species'].unique()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `unique()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `unique()` method on the column `'species'`.
# 
# </details>

# In[71]:


# Find the count of each species type.

penguins.groupby('species').size()


# In[72]:


penguins['species'].value_counts()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `value_counts()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `value_counts()` method on the column `'species'`.
# 
# </details>

# **Question:** How many types of species are present in the dataset?

# 3

# **Question:** Why is it helpful to determine the perfect number of clusters using K-means when you already know how many penguin species the dataset contains?

# To observe at what value of K the clusters become compact and the distance between the points is minimized within the clusters.

# ### Check for missing values

# An assumption of K-means is that there are no missing values. Check for missing values in the rows of the data. 

# In[73]:


# Check for missing values.

penguins.isnull().sum()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `isnull` and `sum` methods. 
# 
# </details>

# Now, drop the rows with missing values and save the resulting pandas DataFrame in a variable named `penguins_subset`.

# In[74]:


# Drop rows with missing values.
# Save DataFrame in variable `penguins_subset\

penguins=penguins.dropna(subset=['sex','body_mass_g','flipper_length_mm','bill_depth_mm','bill_length_mm'])


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `dropna`. Note that an axis parameter passed in to this function should be set to 0 if you want to drop rows containing missing values or 1 if you want to drop columns containing missing values. Optionally, `reset_index` may also be used to avoid a SettingWithCopy warning later in the notebook. 
# </details>

# Next, check to make sure that `penguins_subset` does not contain any missing values.

# In[75]:


# Check for missing values.

penguins.isnull().sum()


# Now, review the first 10 rows of the subset.

# In[76]:


# View first 10 rows.

penguins.head(10)


# ### Encode data
# 
# Some versions of the penguins dataset have values encoded in the sex column as 'Male' and 'Female' instead of 'MALE' and 'FEMALE'. The code below will make sure all values are ALL CAPS. 
# 

# In[77]:


penguins['sex'] = penguins['sex'].str.upper()


# K-means needs numeric columns for clustering. Convert the categorical column `'sex'` into numeric. There is no need to convert the `'species'` column because it isn't being used as a feature in the clustering algorithm. 

# In[78]:


# Convert `sex` column from categorical to numeric.

#converting to categroical columns first
penguins['sex']=penguins['sex'].astype('category')

#now converting to code
penguins['sex']= penguins['sex'].cat.codes
penguins['sex'].unique()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `get_dummies` function. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `drop_first` parameter should be set to `True`. This removes redundant data. The `columns` parameter can **optionally** be set to `['sex']` to specify that only the `'sex'` column gets this operation performed on it. 
# 
# </details>

# ### Drop a column
# 
# Drop the categorical column `island` from the dataset. While it has value, this notebook is trying to confirm if penguins of the same species exhibit different physical characteristics based on sex. This doesn't include location.
# 
# Note that the `'species'` column is not numeric. Don't drop the `'species'` column for now. It could potentially be used to help understand the clusters later. 

# In[79]:


# Drop the island column.

penguins = penguins.drop(columns=['island'],axis=1)
penguins.head(10)


# ### Scale the features
# 
# Because K-means uses distance between observations as its measure of similarity, it's important to scale the data before modeling. Use a third-party tool, such as scikit-learn's `StandardScaler` function. `StandardScaler` scales each point xᵢ by subtracting the mean observed value for that feature and dividing by the standard deviation:
# 
# x-scaled = (xᵢ – mean(X)) / σ
# 
# This ensures that all variables have a mean of 0 and variance/standard deviation of 1. 
# 
# **Note:** Because the species column isn't a feature, it doesn't need to be scaled. 
# 
# First, copy all the features except the `'species'` column to a DataFrame `X`. 

# In[94]:


# Exclude `species` variable from X

X = penguins.drop(columns=['species'])
Standard_Scaler = StandardScaler()
X_scaled = Standard_Scaler.fit_transform(X)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use`drop()`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Select all columns except `'species'.`The `axis` parameter passed in to this method should be set to `1` if you want to drop columns.
# </details>

# Scale the features in `X` using `StandardScaler`, and assign the scaled data to a new variable `X_scaled`. 

# In[95]:


#Scale the features.
#Assign the scaled data to variable `X_scaled`.

X_scaled = Standard_Scaler.fit_transform(X)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Instantiate StandardScaler to transform the data in a single step.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `.fit_transform()` method and pass in the data as an argument.
# </details>

# ## **Step 3: Data modeling** 

# Now, fit K-means and evaluate inertia for different values of k. Because you may not know how many clusters exist in the data, start by fitting K-means and examining the inertia values for different values of k. To do this, write a function called `kmeans_inertia` that takes in `num_clusters` and `x_vals` (`X_scaled`) and returns a list of each k-value's inertia.
# 
# When using K-means inside the function, set the `random_state` to `42`. This way, others can reproduce your results.

# In[107]:


# Fit K-means and evaluate inertia for different values of k.

def kmeans_inertia(num_k, x_scaled):
    inertia=[]
    
    for num in num_k:
        kmeans_ = KMeans(n_clusters = num, random_state=42)
        kmeans_fit = kmeans_.fit(x_scaled)
        inertia.append(kmeans_fit.inertia_)
        
    return inertia


# Use the `kmeans_inertia` function to return a list of inertia for k=2 to 10.

# In[110]:


# Return a list of inertia for k=2 to 10.

inertia = kmeans_inertia(range(2,11), X_scaled)
inertia


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review the material about the `kmeans_inertia` function. 
# </details>

# Next, create a line plot that shows the relationship between `num_clusters` and `inertia`.
# Use either seaborn or matplotlib to visualize this relationship. 

# In[111]:


# Create a line plot.

plot = sns.lineplot(x=range(2,11), y=inertia, marker='o')
plot.set_xlabel('Num_Clusters')
plot.set_ylabel('Inertia')


# 
# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `sns.lineplot`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Include `x=num_clusters` and `y=inertia`.
# </details>

# **Question:** Where is the elbow in the plot?

# the elbow appears at 6 clusters but unable to clearly differentiate the elbow at this point.

# ## **Step 4: Results and evaluation** 

# Now, evaluate the silhouette score using the `silhouette_score()` function. Silhouette scores are used to study the distance between clusters. 

# Then, compare the silhouette score of each value of k, from 2 through 10. To do this, write a function called `kmeans_sil` that takes in `num_clusters` and `x_vals` (`X_scaled`) and returns a list of each k-value's silhouette score.

# In[114]:


# Evaluate silhouette score.
# Write a function to return a list of each k-value's score.

# Fit K-means and evaluate inertia for different values of k.

def kmeans_silhouette(num_k, x_scaled):
    sil_score=[]
    
    for num in num_k:
        kmeans_ = KMeans(n_clusters = num, random_state=42)
        kmeans_fit = kmeans_.fit(x_scaled)
        sil_score.append(silhouette_score(x_scaled, kmeans_fit.labels_))
        
    return sil_score


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review the `kmeans_sil` function video.
# </details>

# Next, create a line plot that shows the relationship between `num_clusters` and `sil_score`.
# Use either seaborn or matplotlib to visualize this relationship. 

# In[121]:



silhouettescore = kmeans_silhouette(range(2,11), X_scaled)
silhouettescore


# In[123]:


# Create a line plot.


import matplotlib.pyplot as plt

sns.lineplot(x=range(2,11), y=silhouettescore, marker='o')
plt.xlabel('Num_Clusters')
plt.ylabel('Silhouette_score')


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `sns.lineplot`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Include `x=num_clusters` and `y=sil_score`.
# </details>

# **Question:** What does the graph show?

# The graph shows extreme edge at 6 clusters hence chhsoing k=6 clusters will be a good choice.

# ### Optimal k-value

# To decide on an optimal k-value, fit a six-cluster model to the dataset. 

# In[125]:


# Fit a 6-cluster model.

kms = KMeans(n_clusters=6, random_state=42)
kms = kms.fit(X_scaled)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Make an instance of the model with `num_clusters = 6` and use the `fit` function on `X_scaled`. 
# </details>
# 
# 
# 

# Print out the unique labels of the fit model.

# In[128]:


# Print unique labels.

np.unique(kms.labels_)


# Now, create a new column `cluster` that indicates cluster assignment in the DataFrame `penguins_subset`. It's important to understand the meaning of each cluster's labels, then decide whether the clustering makes sense. 
# 
# **Note:** This task is done using `penguins_subset` because it is often easier to interpret unscaled data.

# In[133]:


# Create a new column `cluster`.
penguins['cluster'] = kms.labels_
penguins.head(3)


# Use `groupby` to verify if any `'cluster'` can be differentiated by `'species'`.

# In[143]:


# Verify if any `cluster` can be differentiated by `species`.

penguins.groupby(['cluster','species']).size().reset_index()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `groupby(by=['cluster', 'species'])`. 
# 
# </details>
# 

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# 
# Use an aggregation function such as `size`.
# 
# </details>

# Next, interpret the groupby outputs. Although the results of the groupby show that each `'cluster'` can be differentiated by `'species'`, it is useful to visualize these results. The graph shows that each `'cluster'` can be differentiated by `'species'`. 
# 
# **Note:** The code for the graph below is outside the scope of this lab. 

# In[145]:


penguins.groupby(by=['cluster', 'species']).size().plot.bar(title='Clusters differentiated by species',
                                                                   figsize=(6, 5),
                                                                   ylabel='Size',
                                                                   xlabel='(Cluster, Species)');


# Use `groupby` to verify if each `'cluster'` can be differentiated by `'species'` AND `'sex_MALE'`.

# In[147]:


# Verify if each `cluster` can be differentiated by `species' AND `sex_MALE`.

penguins.groupby(['cluster','sex','species']).size().sort_values(ascending=False)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `groupby(by=['cluster','species', 'sex_MALE'])`. 
# </details>
# 

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use an aggregation function such as `size`.
# </details>

# **Question:** Are the clusters differentiated by `'species'` and `'sex_MALE'`?

# [Write your response here. Double-click (or enter) to edit.]

# Finally, interpret the groupby outputs and visualize these results. The graph shows that each `'cluster'` can be differentiated by `'species'` and `'sex_MALE'`. Furthermore, each cluster is mostly comprised of one sex and one species. 
# 
# **Note:** The code for the graph below is outside the scope of this lab. 

# In[149]:


penguins.groupby(by=['cluster','species','sex']).size().unstack(level = 'species', fill_value=0).plot.bar(title='Clusters differentiated by species and sex',
                                                                                                                      figsize=(6, 5),
                                                                                                                      ylabel='Size',
                                                                                                                      xlabel='(Cluster, Sex)')
plt.legend(bbox_to_anchor=(1.3, 1.0))


# ## **Considerations**
# 
# 
# **What are some key takeaways that you learned during this lab? Consider the process you used, key tools, and the results of your investigation.**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 
# **What summary would you provide to stakeholders?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 
# 
# 

# ### References
# 
# [Gorman, Kristen B., et al. “Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis).” PLOS ONE, vol. 9, no. 3, Mar. 2014, p. e90081. PLoS Journals](https://doi.org/10.1371/journal.pone.0090081)
# 
# [Sklearn Preprocessing StandardScaler scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
