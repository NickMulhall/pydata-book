from numpy.random import randn
import numpy as np
np.random.seed(123)
import os
import matplotlib.pyplot as plt
import pandas as pd
import json

pd.options.display.max_rows = 100
df = pd.read_csv('datasets/consumption/360.csv',
  names=['Period', 'Id', 'Type', 'Location', 'RG', 'Instance', 'Cost', 'Currency', 'Country'])
tot = pd.Series([])
for piece in pd:
    print(piece)

