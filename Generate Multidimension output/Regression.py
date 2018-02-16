import numpy as np
import pandas as pd
from sklearn import linear_model

x = np.random.normal(size = 50000)  # generate 50000 numbers from normal distribution

Generated = x[x >= 0]   # delete negatives

X_columns = list(range(0,10))     # X dimension to be 10
y_columns = list(range(0,100))    # y dimension to be 100

X = pd.DataFrame(columns = X_columns)  # created X with 10 coulmns 
y = pd.DataFrame(columns = y_columns)	 # created y with 100 coulumns


#In this part in feeding data
for i in range(1000):
	X.loc[i] = np.random.choice(Generated, 10)   #generating only 10 as our X columns are 10

	y.loc[i] = np.random.choice(Generated, 100)  #generating only 100 as our y columns are 100

#print(y.head(1))


#training part
clf = linear_model.LinearRegression()

clf.fit(X, y)

X_test = pd.DataFrame(columns = X_columns)
X_test.loc[0] = np.random.choice(Generated, 10)

#predicting
w = (clf.predict(X_test.iloc[[0]]))

w = w.reshape((10,10))
print(w)
"""
from matplotlib import pyplot as plt
plt.imshow(w, interpolation='nearest')
plt.show()
"""