# from https://pythonspot.com/k-nearest-neighbors/
# species of plant and classification are Irsis Setosa (0), Iris Versicolour(1) and Iris Virginica (2)
# Plant data only contains classes 0 and 1
import matplotlib
#matplotlib.use('GTKAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd

# Load CSV and columns
df = pd.read_csv("datasets/plants/Plants.csv")
# import some data to play with
iris = datasets.load_iris()

# take the first two features
X = iris.data[:, :2]
y = iris.target

print(X)
