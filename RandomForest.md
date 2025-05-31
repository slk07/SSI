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
print(rf.feature_importances_)
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
