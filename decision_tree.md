# decition tree (simple code)
```
from sklearn import tree

X = [[0, 0], [1, 1]] # input data
Y = [0, 1] # output data or target
clf = tree.DecisionTreeClassifier() # model
clf = clf.fit(X, Y) #train
```
## plot data
```
import matplotlib.pyplot as plt
plt.scatter([x[0] for x in X], [x[1] for x in X], c=Y, cmap='bwr')
plt.show()
```
## plot tree
```
tree.plot_tree(clf)
```
## predict
```
clf.predict([[.51, 2.]])
```
# decision tree (iris data)
```
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)
```
# improving model
cross validation  
evaluation; confusion matrix, accuracy, precision, recall, F1  
```
import pandas as pd
df = pd.read_csv('iris.data', header=None)
print(df.head()) # แสดงข้อมูล 5 แถวแรก
```
