import os 
from os import getcwd, listdir
import zipfile
import tensorflow
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
from zipfile import *
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from cv2 import imread
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


"""### TIME CONSUMING PROCESSING OF IMAGES AND LABELLING THEM"""

image_data = []
image_labels = []
total_classes = 43
height = 224
width = 224
channels = 3
input_path = '‪./Analytics-4/Traffic'
print(getcwd)
images = []
print(os.listdir(input_path))
for i in range(total_classes):
    #path = input_path+r'Train'
    path_file = os.sep.join([input_path ,'Train',str(i)])
    path=path_file
    print(path)
    images = os.listdir(path)
    
    for img in images:
        special=os.sep.join([path,img])
        try:
            print(special)
            image = cv2.imread(special)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((height, width))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error - Image loading")

"""####Converting lists into numpy arrays"""

#Converting lists into numpy arrays
image_data = numpy.array(image_data)
image_labels = numpy.array(image_labels)

dir = 'D:\Analytics-4\Traffic'  
plt.figure(figsize=(10, 10))
for i in range (0,43):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    path = dir + "\meta\{0}.png".format(i)
    img = plt.imread(path)
    plt.imshow(img)
    plt.xlabel(i)

"""####shuffling data"""

#shuffling data
shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

"""###Splitting training and testing dataset"""

#Splitting training and testing dataset
X_train, X_valid, y_train, y_valid = train_test_split(image_data, image_labels, test_size=0.2, random_state=42)#, shuffle=True)

# X_train = X_train/255 
# X_valid = X_valid/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_valid.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_valid.shape)

#Converting the labels into one hot encoding
y_train = tensorflow.keras.utils.to_categorical(y_train, total_classes)
y_valid = tensorflow.keras.utils.to_categorical(y_valid, total_classes)
print(y_train.shape)
print(y_valid.shape)

from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

# Untraining existing weights
for layer in model.layers:
    layer.trainable = False

x = model.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(y_train.shape[1], activation="softmax")(x)
model_final = Model(model.input, predictions)

model_final.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate= 0.0001),loss='binary_crossentropy',metrics=['accuracy'])

eff_history = model_final.fit(x=X_train,y=y_train,batch_size=64, validation_data =(X_valid, y_valid), steps_per_epoch = 100, epochs = 15)

"""### Save the Model"""

model_final.save("traffic_classifier.h5")

"""###PLOT THE GRAPHS"""

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(eff_history.history['accuracy'], label='training accuracy')
plt.plot(eff_history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(eff_history.history['loss'], label='training loss')
plt.plot(eff_history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

pd.DataFrame(eff_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # setting limits for y-axis
plt.show()

"""### Dictionary to Label all the Classes"""

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

"""###TESTING ACCURACY ON TEST DATASET"""

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

test = pd.read_csv('D:\Analytics-4\Traffic\Test.csv')
print(test["ClassId"].values)
print(test["Path"].values)
labels = test["ClassId"].values
imgs = test["Path"].values
print(imgs)
data=[]
for img in imgs:
    image = Image.open((img))
    image = image.resize((30,30))
    data.append(np.array(image))
  
X_valid=np.array(data)
pred1 = model_final.predict(X_valid)
pred = np.argmax(pred1)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
