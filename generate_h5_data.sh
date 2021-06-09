#!/bin/bash

train_list='/home/rash8327/iterative-tsdf-fusion/modelnet_list/train.txt'
test_list='/home/rash8327/iterative-tsdf-fusion/modelnet_list/test.txt'
val_list='/home/rash8327/iterative-tsdf-fusion/modelnet_list/val.txt'

output_root_path_h5='/home/rash8327/Downloads/modelnet_off_h5'

rm -rf $output_root_path_h5
mkdir -p $output_root_path_h5
mkdir -p "$output_root_path/train"
mkdir -p "$output_root_path/test"
mkdir -p "$output_root_path/val"

# create data for training
out_path='$output_root_path/train'
while read line; do
#Reading each line
echo 'Creating depthmaps for train file', $line
python render_depth_from_modelnet.py --in_file $line --out_path $out_path --n_views 100
done < $train_list

# create data for testing
out_path='/home/rajat/Downloads/modelnet_off/test'
while read line; do
#Reading each line
echo 'Creating depthmaps for test file', $line
python render_depth_from_modelnet.py --in_file $line --out_path $out_path --n_views 100
done < $test_list

# create data for validation
out_path='/home/rajat/Downloads/modelnet_off/val'
while read line; do
#Reading each line
echo 'Creating depthmaps for val file', $line
python render_depth_from_modelnet.py --in_file $line --out_path $out_path --n_views 100
done < $val_list

# create raw depth images to train u-net for the depth routing


