import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))
# print(data_dict.keys())
# print(data_dict)

data = np.asarray(data_dict['data'])
labels = np.array(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly!'.format(accuracy * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()


