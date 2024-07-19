import numpy as np
import pandas as pd
import keras
import cv2  
from matpltlib import pyplot as plt
import os
import random 
from PIL import Image
import sklearn











df = pd.read_csv('train.csv')
########## changed to (base_path = "./images")
print(df)

samples = 20000
df = df.loc[:samples, :]
########changed to df = df.loc[df["id"].str.startswith('00',na=false),:]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)

print(num_classes, num_data)

data = pd.DataFrame(df["landmark_id"].value_counts)
data = data.reset_index(inplace=True)
print(data.head())
print(data.tail())










data.columns = ["index", "landmark_id"]

print(data["landmark_id"].describe())

plt.hist(data["landmark_id"], 100, range = (0,58), label = "test")
data = data["landmark_id"].between(0,5).sum()
data = data["landmark_id"].between(5,10).sum()

plt.hist(df["landmark_id"], bins=df["landmark_id"].unique())









#training 
from sklearn import LabelEncoder
lencoder = LabelEncoder()
lencoder.fit(df["landmark_id"])
df.head()

def encode_label(lbl):
    return lencoder.transform(lbl)

def decode_label(lbl):
    return lencoder.inverse_transform(lbl)

def get_image_from_number(num):
    fname, label = df.loc[num, :]
    fname = fname +'jpg'
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    path = os.path.join(f1,f2,f3,fname)
    im = cv2.imread(os.path.join(base_path,path))
    return im, label









print("4 samples images from random classes")
fig = plt.figure(figure=(16,16))
for i in range(1,5):
    ri = random.choices(os.listdir(base_path), k=3)
    folder = base_path + ri[0] + '/' + ri[1] + '/' + ri[2]
    random_img = random_choice(os.listdir(folder))
    img = np.array(Image.open(folder + '/' + random_img))
    fig.add_subplot(1,4,i)
    plt.imshow(img)
    plt.axis('off')
plt.show()









from keras.application.vgg19 import VGG19
from keras.layers import *
from keras import Sequential
tf.compat.v1.disable_eager_execution()

#parameters
learning_rate = 0.001
decay_speed = 1e-6
momentum = 0.09
loss_function = "sparse_categorical_crossentropy"
source_model = VGG(weights=none)
drop_layer1 = Dropout(0.5)
drop_layer2 = Dropout(0.5)

model = Sequential()
for layer in source_model.layers[:-1]:
    if layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Dense(num_classes, activation = "softmax"))
model.summary()

optim1 = keras.optimizer_v1.RMSprop(lr = learning_rate)
model.compile(optimizer = optim1, 
             loss= loss_function,
             metrics = [accuracy])









def image_reshape(in,target_size):
    return cv2.resize(in,target_size)

def get_batch(Dataframe, start, batch_size):
    image_array = []
    label_array = []
    end_img = start + batch_size
    if(end_img)> len(Dataframe):
        end_img = len(Dataframe)
    for idx in range(start, end_img):
        n = idx
        [im, label] = get_image_from_number(0,Dataframe)
        im = image_reshape(im, (224,224)) / 255.0
        image_array.append(im)
        label_array.append(label)

    label_array = encode_label(label_array)
    return np.array(image_array), np, np(label_array)

batch_size = 16
epoch_shuffle = True
weight_classes = True
epochs = 1










#split 
train, val = np.split (df.sample(frac = 1).[int(0.5*len(df))])
print(len(train))
print(len(val))

for e in range(epochs):
    print("Epochs" + str(e+1) + "/" + str(epochs))
    if epoch_shuffle:
        train = train.sample(frac=1)
    for it in range(int(np.cell(len(train)/batch_size))):
    x_train, y_train = get_batch(train, it*batch_size, batch_size)
    model.train_on_batch(x_train,y_train)

model.save("model")









batch_size = 16
errors = 0
good_preds = []
bad_preds = []
for it in range(int(np.cell(len(val)/batch_size))):
    x_val, y_val = get_batch(val, it*batch_size, batch_size)
    result = model.predict(x_val)
    cla = np.argmax(result,axis=1)
    for idx , res in enumerate (result):
        if cla[idx] != y_val[idx]:
            errors += 1
            bad_preds.append([batch_size*it + idx] , cla[idx], res[cla[res]])
        else:
            good_preds.append([batch_size*it + idx] , cla[idx], res[cla[res]])
for i in range(1,6):
    n = int(good_preds[0])
    img, lbl = get_image_from_number(0,val)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.Im_show(img)
