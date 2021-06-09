import random
import numpy as np
import yaml
from easydict import EasyDict


def add_depth_noise(depthmaps, noise_sigma, seed):
    # add noise
    if noise_sigma > 0:
        random.seed(seed)
        np.random.seed(seed)
        sigma = noise_sigma
        #print('add noise with sigma=%f' % sigma)
        noise = np.random.normal(0, 1, size=depthmaps.shape).astype(np.float32)
        depthmaps = depthmaps + noise * sigma * depthmaps
    return depthmaps


def load_yaml(yaml_file_path):
    with open(yaml_file_path, "r") as file:
        yaml_dict = yaml.load(file, yaml.FullLoader)
    return EasyDict(yaml_dict)