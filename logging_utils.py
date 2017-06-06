from tqdm import tqdm

import data


def start_log(filename):
    with open(data.get_log_file(filename), 'w') as f:
        f.write("")
        
def log_print(filename, message):
    tqdm.write(message)
    with open(data.get_log_file(filename), 'a') as f:
        f.write(message + '\n')