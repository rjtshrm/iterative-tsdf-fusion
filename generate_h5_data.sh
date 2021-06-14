#!/bin/bash

#train_list="modelnet_list/train.txt"
#test_list="modelnet_list/test.txt"
#val_list="modelnet_list/val.txt"

output_root_path_h5=$1 # path where to store depth off files
n_views=$2 # number of views
output_root_path_depth=$3 # path where to save depth maps for u-net

#rm -rf $output_root_path_h5 # if already exists
#mkdir -p $output_root_path_h5
#mkdir -p $output_root_path_h5/train
#mkdir -p $output_root_path_h5/test
#mkdir -p $output_root_path_h5/val

rm -rf $output_root_path_depth # if already exists
mkdir -p $output_root_path_depth
mkdir -p $output_root_path_depth/train
mkdir -p $output_root_path_depth/test
mkdir -p $output_root_path_depth/val

# create data for training
out_path=$output_root_path_h5/train
#while read line; do
##Reading each line
#echo 'Creating depthmaps for train file', $line
#python render_depth_from_modelnet.py --in_file $line --out_path $out_path --n_views $n_views
#done < $train_list

# create raw depth images to train u-net for the depth routing
# training
python generate_depth_from_h5.py --in_h5_file_folder $out_path --out_depth_path $output_root_path_depth/train

# create data for testing
out_path=$output_root_path_h5/test
#while read line; do
##Reading each line
#echo 'Creating depthmaps for test file', $line
#python render_depth_from_modelnet.py --in_file $line --out_path $out_path --n_views $n_views
#done < $test_list

# create raw depth images to train u-net for the depth routing
# testing
python generate_depth_from_h5.py --in_h5_file_folder $out_path --out_depth_path $output_root_path_depth/test

# create data for validation
out_path=$output_root_path_h5/val
#while read line; do
##Reading each line
#echo 'Creating depthmaps for val file', $line
#python render_depth_from_modelnet.py --in_file $line --out_path $out_path --n_views $n_views
#done < $val_list

# create raw depth images to train u-net for the depth routing
# validation
python generate_depth_from_h5.py --in_h5_file_folder $out_path --out_depth_path $output_root_path_depth/val



