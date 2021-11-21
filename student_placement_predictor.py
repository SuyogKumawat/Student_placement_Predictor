import numpy as np
import pandas as pd

df = pd.read_csv('students_placement.csv')

X = df.drop(columns=['placed'])
y = df['placed']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
scaler = StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

predicts=rf.predict(np.array([4.5,56,10]).reshape(1,3))
print(predicts)
