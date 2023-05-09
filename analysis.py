# Python script for analysis of Iris Dataset.
# Author: Tom Brophy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read in the dataset
df = pd.read_csv("iris.data",header=None)

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

data = df.drop_duplicates(subset=4)
print(data)