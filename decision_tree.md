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
```
# ตั้งชื่อคอลัมน์ (เพราะไฟล์ไม่มี header)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
print(df.head()) # แสดง 5 แถวแรก
```
## preparing data
missing values  
count class  
slipt X/y  
```
print(df.isnull().sum()) # missing values 
print(df['class'].value_counts()) # # class
X = df.drop('class', axis=1)
y = df['class']

print('X: \n',X)
print('-----------------------------------------------------')
print('y: \n',y)
```
#crossvalidation
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print('X_train:\n', X_train)
print('y_train:\n', y_train)
print('X_test:\n', X_test)
print('y_test:\n', y_test)
```
## create model
```
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', random_state=42) # random_state มีเพื่อเลือก gain หรือ gini ที่เท่ากัน 
model.fit(X_train, y_train)
```
## prediction
```
y_pred = model.predict(X_test)
```
## confusion matrix
```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= y.unique())
disp.plot()
```
## evaluation
```
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=y.unique()))
```
## plot tree
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(50,50))
plot_tree(model, feature_names=X.columns, class_names= y.unique(), filled=True)
plt.show()
```
# Save and load model
install: pip install joblib  
## save model
```
import joblib
joblib.dump(model, 'decision_tree_model.pkl')
```
## load model
```
import joblib
model = joblib.load('decision_tree_model.pkl')

# y_new_pred = model.predict(X_test)
```
# Try:
Wine Dataset  
Breast Cancer Wisconsin   
Zoo Dataset  
Car Evaluation Dataset
