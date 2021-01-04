import pandas as pd
import numpy as np

from utils import train_test_split
from cluster import KMeans

np.random.seed(1234)

iris = pd.read_csv('data/iris.csv')

train, test = train_test_split(iris, train_size=0.7)

x_train = train.drop('Species', axis=1)
x_test = test.drop('Species', axis=1)
y_train = train['Species']
y_test = test['Species']

model = KMeans(k=3, gen_mode='rnd_values')
print(model.fit(x_train))
