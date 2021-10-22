from src.utils.common import read_config
import argparse 
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model,save_plots,pickle_file
import os
import pandas as pd



## Function for Training model
def training (config_path):
    config= read_config(config_path)
    #print(config)


    ## Implementing "get_data" Function from [data_mgmt.py] 
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train),(X_valid, y_valid), (X_test,y_test)= get_data(validation_datasize)


    ## Implementation of "create_model" Function from [model.py]
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]
    model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICS, NUM_CLASSES)


    ## Fitting data into the model
    EPOCHS = config["params"]["epochs"]
    VALIDATION = (X_valid, y_valid)
    history = model.fit(X_train,y_train, epochs=EPOCHS, validation_data= VALIDATION)


   ## IMplementation of ("save_model") Function from [model.py] 
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_name = config["artifacts"]["model_name"]

    model_dir_path = os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path )


    ## IMplementation of ("save_plot") Function from [model.py] 
    plots_dir = config["artifacts"]["plots_dir"]
    plots_name = config["artifacts"]["plots_name"]

    plots_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plots_dir_path,exist_ok = True)
    save_plots(pd.DataFrame(history.history), plots_name, plots_dir_path)


     ## IMplementation of ("pickle_file") Function from [model.py] 
    pickle_model_name= config["artifacts"]["pickle_model_name"]
    pickle_model_dir = config["artifacts"]["pickle_model_dir"]

    pickle_dir_path = os.path.join(artifacts_dir,pickle_model_dir)
    os.makedirs(pickle_dir_path,exist_ok = True)
    pickle_file(model, pickle_model_name, pickle_dir_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path= parsed_args.config)