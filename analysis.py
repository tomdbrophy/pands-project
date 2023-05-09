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

print(df.info)