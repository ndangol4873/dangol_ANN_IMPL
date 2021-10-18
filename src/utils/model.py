import tensorflow as tf


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

    return model_clf ## Untrained model