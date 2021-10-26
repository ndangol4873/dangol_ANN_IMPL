import yaml
import time

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content


## get_timestamp Function
def get_timestamp(name):
    timestamp = time.asctime().replace(" ","_").replace(":","-")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name