
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import pandas as pd 
import pickle as pl


def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS, NUM_CLASSES):

    LAYERS = [
          tf.keras.layers.Flatten(input_shape = [28,28], name= "inputLayer"),## Flattening the 28x28 matrix input in one single array input
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),## Assigning the number of neurons and activation function in the hiddenLayer 1
          tf.keras.layers.Dense(100,activation="relu", name= "hiddenLayer2"),## Assigning the number of neurons and activation function in the hiddenLayer 2
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")## Assigning the number of neurons and activation function in the outputLayer
          ]

    # Model creation "Classifier"
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()

    ## Model compiling
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    return model_clf ## untrained Model

    
    
## Fuction for getting unique_filename
def get_unique_filename(filename):
      unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
      return unique_filename


## Function for saving the model
def save_model(model, model_name, model_dir):
      unique_filename = get_unique_filename(model_name)
      path_to_model = os.path.join(model_dir,unique_filename)
      model.save(path_to_model)


## Function for saving plots
def save_plots(df,plots_name,plots_dir):
      df.plot(figsize=(8,5))
      plt.grid(True)
      plt.gca().set_ylim(0,1)
     
      path_to_plot = os.path.join(plots_dir,plots_name)
      plt.savefig(path_to_plot)
      plt.show()


## Function for saving Pickle file
def pickle_file(model, pickle_model_name, pickle_model_dir):
      unique_file = get_unique_filename(pickle_model_name)
      pickle_model_path = os.path.join(pickle_model_dir,unique_file)
      pl.dump(unique_file, open(pickle_model_path, 'wb'))

     
     


      