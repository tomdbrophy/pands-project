# Python script for analysis of Iris Dataset.
# Author: Tom Brophy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read in the dataset
df = pd.read_csv("iris.data")

# Print top 5 rows
print(df.head())
