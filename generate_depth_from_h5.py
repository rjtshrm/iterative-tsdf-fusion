import h5py as h5
import numpy as np
import argparse
import glob
import cv2


count = 0
def read_depth(h5_file_path, out_path):
    """
    :param h5_file_path:
    :param out_path:
    :return:
    """
    global count
    hd = h5.File(h5_file_path, "r")
    depths = hd.get('depth')[:]
    #pose = hd.get('extrinsics')[:]
    #camera = hd.get('intrinsics')[:]
    for i, d in enumerate(depths):
        cv2.imwrite(f'{out_path}/frame-{count:06d}.depth.png', (d*1000).astype(np.uint16))
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_h5_file_folder", type=str, required=True, help="Input off file folder")
    parser.add_argument("--out_depth_path", type=str, required=True, help="Output save path")

    args = parser.parse_args()
    h5_files = glob.glob(f"{args.in_h5_file_folder}/*")
    for f in h5_files:
        read_depth(f, args.out_depth_path)