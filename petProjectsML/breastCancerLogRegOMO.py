import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pickle

df = pd.read_csv('D:/forPython/PythonProject1/helpFiles/data.csv')
# print(df.keys())

df = df.drop(columns=['Unnamed: 32', 'id'], axis=1)
df.diagnosis = df.diagnosis.map({'M': 1, 'B': 0})

plt.figure(figsize=(7, 5))
sns.countplot(x='diagnosis', data=df, palette='Set2', hue='diagnosis', legend=False)
# plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x=df.index, height=df.diagnosis)
# plt.show()

X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].values

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
sets = [X_train, X_test, y_train, y_test]
for i in range(len(sets)):
    sets[i] = sets[i].T
    print(sets[i].shape, end=' ')
print(f'\n{[s.shape for s in sets]}')

logReg = LogisticRegression()
logReg.fit(X_train, y_train)

y_pred_test = logReg.predict(X_test)
y_pred_train = logReg.predict(X_train)

print(f'train accuracy: {metrics.accuracy_score(y_train, y_pred_train)},'
      f' test accuracy: {metrics.accuracy_score(y_test, y_pred_test)}')
print(f'statistics report: {metrics.classification_report(y_test, y_pred_test)}')

cm = metrics.confusion_matrix(y_test, y_pred_test)
conf_matr = pd.DataFrame(
    data=cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']
)
plt.figure(figsize=(14, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Greens')
plt.show()

with open('D:/forPython/PythonProject1/helpFiles/logRegBreastCan', 'wb') as f:
    pickle.dump(logReg, f)
