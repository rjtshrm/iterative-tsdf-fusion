import h5py as h5
import numpy as np
import torch
import cv2



def read_depth(h5_file_path):
    """
    :param h5_file_path:
    :return:
    """
    hd = h5.File(h5_file_path, "r")
    depths = hd.get('depth')[:]
    pose = hd.get('extrinsics')[:]
    camera = hd.get('intrinsics')[:]
    for i, d in enumerate(depths):
        cv2.imwrite(f'/home/rajat/Desktop/tsdf-gpu/data/frame-{i:06d}.depth.png', (d*1000).astype(np.uint16))
        np.savetxt(f'/home/rajat/Desktop/tsdf-gpu/data/frame-{i:06d}.extrinsic.txt', pose[i, ...])
        np.savetxt(f'/home/rajat/Desktop/tsdf-gpu/data/frame-{i:06d}.intrinsic.txt', camera[i, ...])







if __name__ == "__main__":
    path = "/home/rajat/Desktop/pyrender/data/test.off.h5"
    read_depth(path)
