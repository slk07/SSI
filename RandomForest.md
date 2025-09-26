# decition tree (simple code)
```
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X, y = iris.data, iris.target

# สร้าง Random Forest พร้อมเปิดใช้ OOB
rf = RandomForestClassifier(oob_score=True, random_state=0)
rf.fit(X, y)
```
## ดู OOB accuracy และ feature importance
```
print("OOB accuracy:", rf.oob_score_) # oob acc เฉลี่ย
print(rf.feature_importances_) # Information Gain ถ้าใช้ entropy
for name, importance in zip(iris.feature_names, rf.feature_importances_):
    print(f"{name}: {importance}")
```
## วาดต้นไม้ดูได้
```
from sklearn import tree
import matplotlib.pyplot as plt
# เลือกต้นไม้ต้นแรกใน forest
estimator = rf.estimators_[0] # ปรับค่าตรงนี้ได้

# วาดต้นไม้
plt.figure(figsize=(18,10))
tree.plot_tree(
    estimator,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=12
)
plt.show()
```
# ปรับปรุงmodelให้เหมาะ ใช้การ load file ตรงๆจากในเครื่อง
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
## ตรวจสอบข้อมูล
missing values  
count class  
slipt X/y # cross validation

```
print(df.isnull().sum()) # missing values 
print(df['class'].value_counts()) # # class
X = df.drop('class', axis=1)
y = df['class']

print('X: \n',X)
print('-----------------------------------------------------')
print('y: \n',y)
```
## Cross validation
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
print('X_train:\n', X_train)
print('y_train:\n', y_train)
print('X_test:\n', X_test)
print('y_test:\n', y_test)
```
## create model
```
from sklearn.ensemble import RandomForestClassifier
# สร้าง Random Forest
rf = RandomForestClassifier(

    criterion="entropy", # method
    n_estimators=100,    # ต้นไม้ 100 ต้น
    oob_score=True,      # ใช้ OOB score
    random_state=40      # เพื่อผลลัพธ์ reproducible

)

# เทรนโมเดลด้วย train set
rf.fit(X_train, y_train)

# OOB accuracy (บน train set) ข้อมูลที่บาง tree เคยเห็น เคยtrain
print("OOB accuracy (train):", rf.oob_score_)
```
## predict
```
y_pred = rf.predict(X_test)
```
## ดู feature importance: เฉลี่ยจากต้นไม้แต่ละต้น
```
print("\nFeature Importances:")
for name, importance in zip(X.columns, rf.feature_importances_):
    print(f"{name}: {importance:.4f}")
```
## Confusion matrix
```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= y.unique())
disp.plot()
```
## Evaluations
```
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=y.unique()))
```
# Try: 5 fold
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
