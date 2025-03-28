import os
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

img2vec = Img2Vec()

data_dir = './data/weather'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

data = {}
for j, dir_ in enumerate([train_dir, valid_dir]):
    # print('dir_ : ', dir_)
    features = []
    labels = []
    for category in os.listdir(dir_):
        # print('category_ : ', category)
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_)

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['train_data', 'valid_data'][j]] = features
    data[['train_labels', 'valid_labels'][j]] = labels

model = RandomForestClassifier(random_state=0)
model.fit(data['train_data'], data['train_labels'])

y_pred = model.predict(data['valid_data'])
score = accuracy_score(y_pred, data['valid_labels'])
print(score)

with open('./model.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()

