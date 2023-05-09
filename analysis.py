# Python script for analysis of Iris Dataset.
# Author: Tom Brophy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read in the dataset
df = pd.read_csv("iris.data",header=0, names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

# Print top 5 rows
#print(df.head())

# Outline the shape of the data
#print (df.shape)

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
#sns.pairplot(df.drop([0],axis=0),hue='Species', height=2)
#plt.show(block=True)

