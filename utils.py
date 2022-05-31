
import os #librería para explorar directorios
import time

import pandas as pd #librería para modificar datos tabulares
import numpy as np # Librería de manipulación de matrices con funciones de Álgebra Lineal
import math

import cv2 # Librería de manipulación y visualización de imágenes
#from google.colab.patches import cv2_imshow

from sklearn.utils import shuffle

import matplotlib.pyplot as plt # Librería de manipulación y visualización de imágenes


import tensorflow as tf # Librería para construcción de redes neuronales con funciones propias de Álgebra Lineal
from tensorflow.keras.models import * # Importamos las funciones de inicialización de modelos
from tensorflow.keras.layers import * # Importamos Capas de Redes Neuronales
from tensorflow.keras.optimizers import * # Importamos Optimizadores
from tensorflow.keras.losses import * # Importamos las Funciones de Coste
from tensorflow.keras.utils import Sequence

def make_df(classes):
  """
  Builds pandas DataFrame containing the information about the images (directory, dimensions, label)

  args:
  - classes: list of the different categories in the labels
  """
  image = {}
  size = {}
  label = {}
  counter = 0
  for clas in classes:
    cat = os.path.join('data', clas)
    for foto in os.listdir(cat):
      full_path = os.path.join(cat, foto)
      counter += 1

      image[counter] =  full_path
      label[counter] = clas
      size[counter] = cv2.imread(full_path).shape[:2]

  list_dicts = [image, size, label]

  df = pd.DataFrame(list_dicts).transpose()
  df.rename(columns = {0: 'image', 1:'Size', 2: 'label'}, inplace = True)

  return df

class generator(Sequence):
  
  def __init__(self, df, im_size, batch_size):
    """
    Método constructor de la clase 'generator', permite generar las variables de interés que
    compartirán todos los métodos

    - df : dataframe to be used
    - x, y : images and their corresponding labels
    - im_size : tuple containing the dimensions which we want to rescale the images to
    - batch_size : size of the batch returned by the '__getitem__' method
    """
    self.df = df
    self.x, self.y = df.image, df.label
    self.im_size = im_size
    self.batch_size = batch_size

  def __len__(self):
    return math.ceil(len(self.x)/self.batch_size)

  def create_label(self):
    """
    Returns a matrix with the label vectors
    """
    n_classes = len(self.y.unique())
    a = np.array(list(range(n_classes)))
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1

    return b

  def get_label(self, label_matrix, clase):
    """
    Retorna el vector correspondiente de la matriz generada en la función 'create_label'
    """
    return label_matrix[int(clase) -1 ]
  
  def __getitem__(self, idx):
    """
    Returns the images tensors and their labels
    """
    label_matrix = self.create_label()

    batch_x = self.x[idx * self.batch_size:(idx + 1) *
      self.batch_size]
    batch_y = self.y[idx * self.batch_size:(idx + 1) *
      self.batch_size]

    X = [cv2.resize(cv2.imread(x, cv2.IMREAD_GRAYSCALE)/255., self.im_size) for x in batch_x]
    
    int_y = [int(y) for y in batch_y]
    batch_Y = [self.get_label(label_matrix, y) for y in int_y]

    
    X = tf.cast(X, dtype = tf.float32)
    Y = tf.cast(batch_Y, dtype = tf.float16)
    return X, Y
    
  def view_image(self, index):
    """
    Allows us to visualize the image of a given index
    """
    img = cv2.resize(cv2.imread(self.x[index], cv2.IMREAD_GRAYSCALE), self.im_size)
    plt.title('label = ' + str(self.y[index]))
    plt.imshow(img, interpolation = 'none', cmap = 'gray')

def train_val_test(df, train_size = 0.7, validation = True):
  df_shuffled = shuffle(df)

  # Establecemos los índices para segmentar los datos
  train_index = int(len(df_shuffled)*train_size)

  if validation:
    val_index = int(len(df_shuffled)*(train_size + (1-train_size)/2))
    
    df_train = df_shuffled.iloc[:train_index]
    df_val = df_shuffled.iloc[train_index:val_index]
    df_test = df_shuffled.iloc[val_index:]

    return df_train, df_val, df_test
  
  else:
    df_train = df_shuffled.iloc[:train_index]
    df_test = df_shuffled.iloc[train_index:]

    return df_train, df_test

def cnn_1(learning_rate = 1e-4, input_shape = (300, 300, 1)):
  inputs = Input(shape = input_shape)

  x = BatchNormalization()(inputs)
  x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu')(inputs)
  x = MaxPooling2D(pool_size = (2,2))(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)

  #x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
  #x = MaxPooling2D(pool_size = (2,2))(x)
  #x = BatchNormalization()(x)
  #x = Dropout(0.3)(x)

  #x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
  #x = MaxPooling2D(pool_size = (2,2))(x)
  #x = BatchNormalization()(x)
  #x = Dropout(0.3)(x)

  x = Flatten()(x)
  #x = Dense(256, activation = 'relu')(x)
  #x = Dropout(0.5)(x)
  #x = Dense(128, activation = 'relu')(x)
  #x = Dropout(0.5)(x)
  x = Dense(32, activation = 'relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(3, activation = 'sigmoid')(x)
  
  model = Model(inputs = inputs, outputs = x)

  print('Compilando modelo...')
  model.compile(loss = CategoricalCrossentropy(from_logits = False),
                optimizer = Adam(learning_rate = learning_rate),
                metrics = ['accuracy'])
  print('¡Modelo compilado!')

  return model

def train_model(model, df_train , df_val, im_size, explore_lr = False, epochs = 25):
  """
  Trains the defined cnn
  
  Args:
  - model: neural network previously defined
  - df_train: dataframe containing the info about the training samples
  - df_val: dataframe containing the info about the validation samples
  - im_size: dimensions to which we want the images rescaled
  - explore_lr (boolean): if True, it varies the learning rate throughout the epochs and lets us visualize its optimal value
  - learning_rate: learning rate
  - epochs: number of times the model will iterate through the training data
  """
  # Para monitorizar el ratio de aprendizaje y controlar el entrenamiento
  lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*5**(epoch/20))
  earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 15, mode = 'max', restore_best_weights=True)

  if explore_lr == True:
    callbacks = [earlystopping, lr]

  else:
    callbacks = [earlystopping]

  history = model.fit(x = generator(batch_size = 64, df = df_train, im_size = im_size),
                    validation_data = generator(batch_size = 64, df = df_val, im_size = im_size),  
                    epochs=epochs,
                    # steps_per_epoch = 100,
                    verbose = 1,
                    callbacks = callbacks
                    )
  
  return history

# Matriz de confusión bonita

def pretty_confusion_matrix(y_true, y_pred, classes = None, figsize = (10, 10),
                          text_size = 15):
  """
  Represents the confusion matrix of the classification proces carried out
  
  Args:

  y_true: real label
  y_pred: binarized predictions
  classes: list containing the classes
  figsize: tuple indicating the dimensions which the images will be rescaled to
  text_size: font size
  """

  import itertools
  from sklearn.metrics import confusion_matrix

  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype('float') / cm.sum(axis = 1) [:, np.newaxis]
  n_classes = cm.shape[0]
  
  fig, ax = plt.subplots(figsize = figsize)
  cax = ax.matshow(cm, cmap = plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  ax.set(title = 'Confusion Matrix',
         xlabel = 'Predicted label',
         ylabel = 'Real label',
         xticks = np.arange(n_classes),
         
         yticks = np.arange(n_classes),
         xticklabels = labels,
         yticklabels = labels)
  
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  threshold = (cm.max() + cm.min()) /2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]*100:.1f})%',
             horizontalalignment = 'center', 
             color = 'white' if cm[i, j] > threshold else 'black', 
             size = text_size)
