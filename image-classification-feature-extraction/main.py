import os
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

img2vec = Img2Vec()

data_dir = './data/weather'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    print('dir_ : ', dir_)
    features = []
    labels = []
    for category in os.listdir(dir_):
        print('category_ : ', category)
        for img_path in os.listdir(os.path.join(dir_, category)):
            print(img_path)