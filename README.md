# pands-project
Repository for Programming and Scripting project analyzing the Iris Dataset.

# Background
The Iris Dataset is a dataset published by R.A. Fisher in 1936 [^1] . The data has been widely used for studying statistics and machine learning.

The iris dataset contains measurements on three species of iris flowers. The measurements are:
- Sepal length
- Sepal width
- Petal length
- Petal width

All measurements were taken in cm. In addition to the 4 measurements, the data also records the species of each entry.
There are a total of 150 data entries in the database. This is made up of 50 entries per species.

# Analysis
Assistance for initial exploratory analysis of the iris dataset can be found in multiple locations online [^2] [^3].

There are multiple sources to get the iris dataset online and there are some slight variations in the dataset depending on where it is acquired. The data itself should be unaltered in each source but some versions contain headings for columns, an ID column, or a number corresponding to the species (0, 1, or 2) [^2] [^4]. 

Initial analysis confirms the shape of the data (150, 5), meaning 150 rows of data and 5 columns. This is as expected having sourced the data from the UCI Machine Learning Repository [^1]. Using DataFrame.isnull() also confirms that there are no null values, so there is no missing data.

Running some initial statistical analysis using DataFrame.describe() show us the following broad information about the data:
| | Sepal Length | Sepal Width | Petal Length | Petal Width |
|---|---|---|---|---|
|Mean| 5.843 | 3.054 | 3.759 | 1.199 |
|Std| 0.828 | 0.434 | 1.764 | 0.763 |


[^1]: https://archive.ics.uci.edu/ml/datasets/iris
[^2]: https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/
[^3]: https://medium.com/analytics-vidhya/exploratory-data-analysis-iris-dataset-4df6f045cda
[^4]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
