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
