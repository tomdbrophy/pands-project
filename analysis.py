# Python script for analysis of Iris Dataset.
# Author: Tom Brophy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

f.close()


#Information about the dataset 
#print(df.info)

# Gives info about columns such as mean values and standard deviation
#print(df.describe())

# Checks for missing values
#print(df.isnull().sum())

# Checks for duplicate values in species column. Only 3 unique values
#data = df.drop_duplicates(subset=4)
#print(data)

# Counts the number of each species recorded
#print(df.value_counts([4]))

# Produces a scatter plot of Sepal Length vs Sepal Width
#sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
#plt.legend()
#plt.show()

# Produces a scatter plot of Petal Length vs Petal Width
#sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species')
#plt.legend()
#plt.show()

# Produces a set of scatter plots comparing each pair of variables
sns.pairplot(df.drop([0],axis=0),hue='Species', height=2)
plt.savefig('scatterplot_array.png')
plt.close()

# Produces a set of histograms for each variable
'''
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes[0,0].set_title('Sepal Length')
axes[0,0].hist(df['SepalLengthCm'], bins=10)

axes[0,1].set_title('Sepal Width')
axes[0,1].hist(df['SepalWidthCm'], bins=10)

axes[1,0].set_title('Petal Length')
axes[1,0].hist(df['PetalLengthCm'], bins=10)

axes[1,1].set_title('Petal Width')
axes[1,1].hist(df['PetalWidthCm'], bins=10)

plt.show()
'''
# Calculates pairwise correlations between variables
#print(df.corr(method='pearson'))

# Create heatmaps based on pairwise correlations
#sns.heatmap(df.corr(method='pearson'), annot=True)
#plt.show()

# Box plot by species and variable
'''
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

plt.show()
'''

