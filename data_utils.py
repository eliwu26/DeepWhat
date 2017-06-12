import numpy as np
from PIL import Image


def get_spatial_features(example, obj):
    img_width, img_height = example['image']['width'], example['image']['height']
    
    x, y, bbox_width, bbox_height = obj['bbox']
    x = 2 * (x / img_width) - 1
    y = 2 * (y / img_height) - 1
    bbox_width = 2 * bbox_width / img_width
    bbox_height = 2 * bbox_height / img_height

    return (x, y, x + bbox_width, y + bbox_height, x + bbox_width / 2, y + bbox_width / 2, bbox_width, bbox_height)

def img_from_path(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.load()
    return np.array(img)