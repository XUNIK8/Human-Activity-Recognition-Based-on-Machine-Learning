import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


inputs = pd.read_csv('../myFeature/Android_feature_1341_54_Final.csv', header=None)
inputs = shuffle(inputs)
trainx = np.array(inputs.iloc[:1340, :54])
trainy = np.array(inputs.iloc[:1340, 54])
trainy = trainy.ravel()

rfc = RandomForestClassifier(n_estimators=100,n_jobs = -1,max_features = "auto")
model = rfc.fit(trainx,trainy)
recall_score = cross_val_score(rfc, trainx, trainy, cv=10)

print('RF:')
print(recall_score)
print(np.mean(recall_score))

knn = KNeighborsClassifier()  # 引入训练方法
recall_score1 = cross_val_score(knn, trainx, trainy, cv=10)
print('KNN:')
print(recall_score1)
print(np.mean(recall_score1))


joblib.dump(model, '../myModel/Android_test_model_RF_98.1_5Actions.model')
joblib.dump(knn, '../myModel/Android_test_model_KNN_89_5Actions.model')