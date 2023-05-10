# Python script for analysis of Iris Dataset.
# Author: Tom Brophy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import lognorm

# Read in the dataset
df = pd.read_csv("iris.data",header=None, names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

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

f.write('\n\nBelow shows the shape of the data:\n')
data_shape = df.shape
data_shape_string = str(data_shape)
f.write(data_shape_string)
f.write('\nThis tells us that there are 150 rows of data and 5 columns.\n\n')

f.write('\nBelow gives some statistical information about the data in each column:\n\n')
data_described = df.describe()
data_described_string = data_described.to_string()
f.write(data_described_string)

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

# Separate out data by Species
setosa_df = df[df['Species'] == 'Iris-setosa']
versicolor_df = df[df['Species'] == 'Iris-versicolor']
virginica_df = df[df['Species'] == 'Iris-virginica']

# Check to see if data is normally distributed
def norm_check (data):
    stat, p_val = shapiro(data)
    if p_val < 0.5:
        return 'Not normally distributed'
    else:
        return 'Normally distributed'


sepal_length_setosa = setosa_df['SepalLengthCm']
norm_sep_len_set = norm_check(sepal_length_setosa)
f.write(f'\n\nThe sepal length of Iris Setosa in this dataset is:\n{norm_sep_len_set}')



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
