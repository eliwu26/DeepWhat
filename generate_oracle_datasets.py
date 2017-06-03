import pickle
import json

# load pytorch etc here

import data

# add functions to get processed features here

if __name__ == '__main__':
    # load pretrained model here
    
    for split in ('train', 'valid', 'test'):
        with open(data.get_gw_file(split), 'r') as f:
            for line in f:
                example = json.loads(line)