#Competition: https://www.kaggle.com/datasets/andrewmvd/leukemia-classification
#pip install -q --upgrade keras-cv
#pip install -q --upgrade keras

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import keras
from keras import optimizers
import keras_cv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras_cv.models import ImageClassifier
import matplotlib.pyplot as plt


def create_df(dataset):
    image_paths, labels = [], []
    for dirpath, dirnames, filenames in os.walk(dataset):
        for filename in filenames:
            image = os.path.join(dirpath, filename)
            image_paths.append(image)
            if dirpath[-3:] == 'all':
                labels.append('all')
            else:
                labels.append('hem')
    df = pd.DataFrame({'Image Path': image_paths,
                       'Label': labels})
    return df


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

train_dir = "C:/Users/Lenovo/Desktop/Lukemia Classification/archive/C-NMC_Leukemia/training_data"
df =  create_df(train_dir)

train_df, valid_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=31, stratify=df['Label'])

train_data_generator = ImageDataGenerator(horizontal_flip=True)
valid_data_generator = ImageDataGenerator()

BATCH_SIZE = 32

train_gen = train_data_generator.flow_from_dataframe( train_df, x_col= 'Image Path', y_col= 'Label', target_size= (224, 224), class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= BATCH_SIZE)

valid_gen = valid_data_generator.flow_from_dataframe( valid_df, x_col= 'Image Path', y_col= 'Label', target_size= (224, 224), class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= BATCH_SIZE)

train_steps = train_gen.n // train_gen.batch_size + 1
validation_steps = valid_gen.n // valid_gen.batch_size


model = keras_cv.models.ImageClassifier.from_preset("efficientnetv2_b0_imagenet", num_classes=2)

model.load_weights('model.h5')

YV = train_df['Label'].map({'all':0, 'hem':1}).values
y_pred = model.predict(train_gen)
#print(YV.shape) --> (615,)
#print(y_pred.shape) --> (615, 2)
y_pred = np.argmax(y_pred,axis  =1)
#print(y_pred.shape) --> (615,)
A = accuracy_score(y_true = YV, y_pred = y_pred)

print('Accuracy = ', str(A))
print(confusion_matrix(y_true = YV, y_pred = y_pred))
print(classification_report(y_true = YV, y_pred = y_pred, labels = [0,1]))


