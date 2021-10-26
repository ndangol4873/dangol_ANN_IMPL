import tensorflow as tf
import numpy as np
import time 
import os
from src.utils.common import get_timestamp


## callbacks Functions
def callbacks(config, X_train):
    logs = config["logs"]
    unique_dir_name = get_timestamp("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs["logs_dir"], logs["TENSORBOARD_ROOT_LOG_DIR"], unique_dir_name)
    
    ## Making TENSORBOARD_ROOT_LOGS_DIR
    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True) 


    ## Creating tensorboard callback object with log_dir path       
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir= TENSORBOARD_ROOT_LOG_DIR)


    ## Creating filewiter objects which writes the TENSORBOAR LOGS
    file_writer = tf.summary.create_file_writer(logdir = TENSORBOARD_ROOT_LOG_DIR)


    ## Write the images along with the TENSORBOARD_LOGS
    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handritten digit samples", images, max_outputs=25, step=0)


    ## Creating early_stopping_Callbacks
    params = config["params"]
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=params["patience"], 
        restore_best_weights= params["restore_best_weights"]) ## Calling callbacks built with tf.keras


    ## Creating the Checkpoint Callbacks
    artifacts = config["artifacts"]
    CKPT_dir = os.path.join(artifacts["artifacts_dir"], artifacts["CHECKPOINT_DIR"])
    os.makedirs(CKPT_dir, exist_ok=True)

    CKPT_path = os.path.join(CKPT_dir, "model_ckpt.h5")

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True) ## calling callbacks built with tf.keras

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]
