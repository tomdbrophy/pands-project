# Python script for analysis of Iris Dataset.
# Author: Tom Brophy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import lognorm
from sklearn.model_selection import train_test_split

# Read in the dataset
df = pd.read_csv("iris.data",header=None, names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

# Separate out data by Species
setosa_df = df[df['Species'] == 'Iris-setosa']
versicolor_df = df[df['Species'] == 'Iris-versicolor']
virginica_df = df[df['Species'] == 'Iris-virginica']

# Histogram for Sepal Length
plt.hist(df['SepalLengthCm'], bins=10)
plt.savefig('sepal_length_histogram.png')
plt.close()

# Histogram for Sepal Width
plt.hist(df['SepalWidthCm'], bins=10)
plt.savefig('sepal_width_histogram.png')
plt.close()

# Histogram for Petal Length
plt.hist(df['PetalLengthCm'], bins=10)
plt.savefig('petal_length_histogram.png')
plt.close()

# Histogram for Petal Width
plt.hist(df['PetalWidthCm'], bins=10)
plt.savefig('petal_width_histogram.png')
plt.close()

# This section deals with outputting summary information to a text file.
f = open('iris_variable_summary.txt', 'w')
f.write('This file will contain information summarising variables in the Iris Dataset.\n\n')
f.write('Below shows the head of the data file.\nThis includes the column headings and top 5 rows of data.\n\n')
data_head = df.head()
data_head_string = data_head.to_string()
f.write(data_head_string)

# Shape
f.write('\n\nBelow shows the shape of the data:\n')
data_shape = df.shape
data_shape_string = str(data_shape)
f.write(data_shape_string)
f.write('\nThis tells us that there are 150 rows of data and 5 columns.\n\n')

# Statistical Description
f.write('\nBelow gives some statistical information about the data in each column:\n\n')
data_described = df.describe()
data_described_string = data_described.to_string()
f.write(data_described_string)

# Statistical description broken down by species
f.write('\n\nFurther statistical info broken down by species.')
f.write('\nIris Setosa:\n')
data_desc_set = setosa_df.describe()
data_desc_set_str = data_desc_set.to_string()
f.write(data_desc_set_str)
f.write('\nIris Versicolor:\n')
data_desc_vers = versicolor_df.describe()
data_desc_vers_str = data_desc_vers.to_string()
f.write(data_desc_vers_str)
f.write('\nIris Virginica:\n')
data_desc_virg = virginica_df.describe()
data_desc_virg_str = data_desc_virg.to_string()
f.write(data_desc_virg_str)

# Checks for null values
f.write('\n\nBelow gives information on whether any columns contain null values:\n\n')
data_null = df.isnull().sum()
f.write(str(data_null))
f.write('\nWe can see from the above that there are no missing values in the dataset.\n')

f.write('\n\nBelow drops duplicates to confirm the number of unique values in the Species Column.\n\n')
data_uniques = df.drop_duplicates(subset='Species')
data_uniques_string = data_uniques.to_string()
f.write(data_uniques_string)
f.write('\n\nFrom the above we can see that there are only 3 unique values for Species in the dataset.\n')

f.write('\n\nBelow counts the number of entries in the dataset for each Species:\n\n')
data_counts = df.value_counts(['Species'])
f.write(str(data_counts))
f.write('\nFrom the above we can see that there is an equal number (50) of entries for each Species.\n\n')

f.write('\n\nBelow figures show pairwise correlations between variables:\n\n')
data_corr = df.corr(method='pearson')
#data_corr_string = data_corr.to_string
f.write(str(data_corr))



# Check to see if data is normally distributed
# Shapiro checks for normal distribution. The below function then gives a binary response based on p-value being over or under 0.05
def norm_check (data):
    stat, p_val = shapiro(data)
    if p_val < 0.05:
        return f'Not normally distributed. P-Value = {p_val}'
    else:
        return f'Normally distributed. P-Value = {p_val}'

# Normality check for sepal length of all species combined.
norm_sep_len = norm_check(df['SepalLengthCm'])
f.write(f'\n\nThe sepal length for all sepcies in the dataset combined is:\n{norm_sep_len}')

# Normality checks for sepal length of each species.
norm_sep_len_set = norm_check(setosa_df['SepalLengthCm'])
f.write(f'\n\nThe sepal length of Iris Setosa in this dataset is:\n{norm_sep_len_set}')

norm_sep_len_vers = norm_check(versicolor_df['SepalLengthCm'])
f.write(f'\n\nThe sepal length of Iris Versicolor in this dataset is:\n{norm_sep_len_vers}')

norm_sep_len_virg = norm_check(virginica_df['SepalLengthCm'])
f.write(f'\n\nThe sepal length for Iris Virginica in this dataset is:\n{norm_sep_len_virg}')

# Normality check for sepal width of all species combined.
norm_sep_wid = norm_check(df['SepalWidthCm'])
f.write(f'\n\nThe sepal width for all species combined is:\n{norm_sep_wid}')

# Normality check for sepal width of each species.
norm_sep_wid_set = norm_check(setosa_df['SepalWidthCm'])
f.write(f'\n\nThe sepal width for Iris Setosa in this datset is:\n{norm_sep_wid_set}')

norm_sep_wid_vers = norm_check(versicolor_df['SepalWidthCm'])
f.write(f'\n\nThe sepal width for Iris Versicolor in this dataset is:\n{norm_sep_wid_vers}')

norm_sep_wid_virg = norm_check(virginica_df['SepalWidthCm'])
f.write(f'\n\nThe sepal width for Iris Virginica in this dataset is:\n{norm_sep_wid_virg}')

# Normality check for petal length of all species combined.
norm_pet_len = norm_check(df['PetalLengthCm'])
f.write(f'\n\nThe petal length for all species in this dataset combined is:\n{norm_pet_len}')

# Normality check for petal length of individual species.
norm_pet_len_set = norm_check(setosa_df['PetalLengthCm'])
f.write(f'\n\nThe petal length for Iris Setosa in this dataset is:\n{norm_pet_len_set}')

norm_pet_len_vers = norm_check(versicolor_df['PetalLengthCm'])
f.write(f'\n\nThe petal length for Iris Versicolor in this dataset is:\n{norm_pet_len_vers}')

norm_pet_len_virg = norm_check(virginica_df['PetalLengthCm'])
f.write(f'\n\nThe petal length for Iris Virginica in this datset is:\n{norm_pet_len_virg}')

# Normality check for Petal width of all species combined.
norm_pet_wid = norm_check(df['PetalWidthCm'])
f.write(f'\n\nThe petal width for all species in this datset combined is:\n{norm_pet_wid}')

# Normality check for petal width of individual species.
norm_pet_wid_set = norm_check(setosa_df['PetalWidthCm'])
f.write(f'\n\nThe petal width for Iris Setosa in this dataset is:\n{norm_pet_wid_set}')

norm_pet_wid_vers = norm_check(versicolor_df['PetalWidthCm'])
f.write(f'\n\nThe petal width for Iris Versicolor in this dataset is:\n{norm_pet_wid_vers}')

norm_pet_wid_virg = norm_check(virginica_df['PetalWidthCm'])
f.write(f'\n\nThe petal width for Iris Virginica in this datset is:\n{norm_pet_wid_virg}')

f.close()


# Produces a scatter plot of Sepal Length vs Sepal Width
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
plt.savefig('sepal_scatter.png')
plt.close()

# Produces a scatter plot of Petal Length vs Petal Width
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species')
plt.savefig('petal_scatter.png')
plt.close()

# Produces a set of scatter plots comparing each pair of variables
sns.pairplot(df.drop([0],axis=0),hue='Species', height=2)
plt.savefig('scatterplot_array.png')
plt.close()

# Produces a set of histograms for each variable
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes[0,0].set_title('Sepal Length')
axes[0,0].hist(df['SepalLengthCm'], bins=10)

axes[0,1].set_title('Sepal Width')
axes[0,1].hist(df['SepalWidthCm'], bins=10)

axes[1,0].set_title('Petal Length')
axes[1,0].hist(df['PetalLengthCm'], bins=10)

axes[1,1].set_title('Petal Width')
axes[1,1].hist(df['PetalWidthCm'], bins=10)

plt.savefig('grouped_histograms.png')
plt.close()

# Produces a set of histograms for each variable for the Iris Setosa
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes[0,0].set_title('Sepal Length')
axes[0,0].hist(setosa_df['SepalLengthCm'], bins=10)

axes[0,1].set_title('Sepal Width')
axes[0,1].hist(setosa_df['SepalWidthCm'], bins=10)

axes[1,0].set_title('Petal Length')
axes[1,0].hist(setosa_df['PetalLengthCm'], bins=10)

axes[1,1].set_title('Petal Width')
axes[1,1].hist(setosa_df['PetalWidthCm'], bins=10)

fig.suptitle('Iris Setosa')
plt.savefig('grouped_histograms_setosa.png')
plt.close()

# Produces a set of histograms for each variable for the Iris Versicolor
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes[0,0].set_title('Sepal Length')
axes[0,0].hist(versicolor_df['SepalLengthCm'], bins=10)

axes[0,1].set_title('Sepal Width')
axes[0,1].hist(versicolor_df['SepalWidthCm'], bins=10)

axes[1,0].set_title('Petal Length')
axes[1,0].hist(versicolor_df['PetalLengthCm'], bins=10)

axes[1,1].set_title('Petal Width')
axes[1,1].hist(versicolor_df['PetalWidthCm'], bins=10)

fig.suptitle('Iris Versicolor')
plt.savefig('grouped_histograms_versicolor.png')
plt.close()

# Produces a set of histograms for each variable for Iris Virginica
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes[0,0].set_title('Sepal Length')
axes[0,0].hist(virginica_df['SepalLengthCm'], bins=10)

axes[0,1].set_title('Sepal Width')
axes[0,1].hist(virginica_df['SepalWidthCm'], bins=10)

axes[1,0].set_title('Petal Length')
axes[1,0].hist(virginica_df['PetalLengthCm'], bins=10)

axes[1,1].set_title('Petal Width')
axes[1,1].hist(virginica_df['PetalWidthCm'], bins=10)

fig.suptitle('Iris Virginica')
plt.savefig('grouped_histograms_virginica.png')
plt.close()

# Create heatmap based on pairwise correlations
sns.heatmap(df.corr(method='pearson'), annot=True)
plt.savefig('correlation_heatmap.png')
plt.close()

# Box plot by species and variable
def graph(y):
    sns.boxplot(x='Species', y=y, data=df)

plt.figure(figsize=(10,10))

plt.subplot(221)
graph('SepalLengthCm')

plt.subplot(222)
graph('SepalWidthCm')

plt.subplot(223)
graph('PetalLengthCm')

plt.subplot(224)
graph('PetalWidthCm')

plt.savefig('box_plots.png')
plt.close()

# Sepal Length Kernel Density Estimation
sns.kdeplot(data=df, x='SepalLengthCm', hue='Species')
plt.savefig('sepal_length_kde.png')
plt.close()

# Sepal Width Kernel Density Estimation
sns.kdeplot(data=df, x='SepalWidthCm', hue='Species')
plt.savefig('sepal_width_kde.png')
plt.close()

# Petal Length Kernel Density Estimation
sns.kdeplot(data=df, x='PetalLengthCm', hue='Species')
plt.savefig('petal_length_kde.png')
plt.close()

# Petal Width Kernel Density Estimation
sns.kdeplot(data=df, x='PetalWidthCm', hue='Species')
plt.savefig('petal_width_kde.png')
plt.close()


