import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

plt.rcParams['figure.figsize']=(16, 9)
plt.style.use('ggplot')

kmeans_data = pd.read_csv('kmeans_data.csv', header=None)

x1 = kmeans_data.iloc[:,[0,1]].values



