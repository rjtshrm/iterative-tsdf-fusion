import glob
import random
from torch.utils.data import Dataset
import cv2 as cv
import utils


class DepthDataset(Dataset):
    def __init__(self, data_dir, noise_range=[0.005, 0.01, 0.02, 0.03]):
        self.data_dir = data_dir
        self.depth_files = None
        self.noise_range = noise_range
        self.read_data_dir()

    def read_data_dir(self):
        depths = []
        self.depth_files = glob.glob(f"{self.data_dir}/*")

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, index):
        depth_file = self.depth_files[index]
        depth = cv.imread(f"{depth_file}", -1).astype('float') / 1000.  # in meters
        depth_with_noise = utils.add_depth_noise(depth, random.choice(self.noise_range), index)
        data = {'noisy_depth': depth_with_noise, 'depth': depth}
        return data
