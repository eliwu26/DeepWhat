import pickle
import numpy as np

import data

for split in ['train', 'valid', 'test']:
    for small in [True, False]:
        with open(data.get_processed_file('oracle', split, small), 'rb') as f:
            a = pickle.load(f)
        b = a[2].astype(np.float32)
        with open(data.get_processed_file('oracle', split, small), 'wb') as f:
            pickle.dump((a[0], a[1], b, a[3], a[4]), f, protocol=4)
