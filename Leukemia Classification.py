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
                                    color_mode= 'rgb', shuffle= True, batch_size= BATCH_SIZE)

valid_gen = valid_data_generator.flow_from_dataframe( valid_df, x_col= 'Image Path', y_col= 'Label', target_size= (224, 224), class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= BATCH_SIZE)

train_steps = train_gen.n // train_gen.batch_size + 1
validation_steps = valid_gen.n // valid_gen.batch_size


model = keras_cv.models.ImageClassifier.from_preset("efficientnetv2_b0_imagenet", num_classes=2)

model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"])

epochs = 20

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=valid_gen,
    validation_steps=validation_steps,
    epochs=epochs,
    batch_size=BATCH_SIZE,
    verbose=1)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

x = int(input('Do you like it? [write 1 if you do]'))
if x == 1:
    model.save('model.h5')
#model = keras.models.load_model('model.h5')

YV = valid_df['Label'].values
y_pred = model.predict(valid_gen)
print(YV.shape)
print(y_pred.shape)
"""A = accuracy_score(y_true = YV, y_pred = y_pred)

print('Accuracy = ', str(A))
print(confusion_matrix(y_true = YV, y_pred = y_pred))
print(classification_report(y_true = YV, y_pred = y_pred, labels = [0,1]))"""

"""Ressources:
https://www.ibm.com/topics/computer-vision#:~:text=Computer%20vision%20is%20a%20field,recommendations%20based%20on%20that%20information.
https://www.v7labs.com/blog/what-is-computer-vision
https://history-computer.com/computer-scanner/#:~:text=A%20team%20led%20by%20Russell%20Kirsch%20was%20the%20inventor%20of,image%20scanned%20on%20this%20scanner.
https://medium.com/@boelsmaxence/introduction-to-image-processing-filters-179607f9824a
https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
https://www.ibm.com/topics/convolutional-neural-networks
https://dev.to/ruthvikraja_mv/mathematical-formulae-behind-optimization-algorithms-for-neural-networks-121p
"""


