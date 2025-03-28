import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = '/content/data/valid/cloudy/cloudy118_jpg.rf.afeedcffe421a88fff0b764a091a1252.jpg'

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])

print(pred)