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
