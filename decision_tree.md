# decition tree (simple code)
```
from sklearn import tree

X = [[0, 0], [1, 1]] # input data
Y = [0, 1] # output data or target
clf = tree.DecisionTreeClassifier() # model
clf = clf.fit(X, Y) #train
```
## ploting data
```
import matplotlib.pyplot as plt
plt.scatter([x[0] for x in X], [x[1] for x in X], c=Y, cmap='bwr')
plt.show()
```
