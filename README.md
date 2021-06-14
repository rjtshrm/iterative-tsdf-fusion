### Dataset Preparation:
Inside the `modelnet_list` directory, run `modelnet_list_generator.py`. `modelnet_list_generator.py`
expects different args which are as

- modelnet_data_path
- train_num
- test_num
- val_num

Running the above file will generate train, test and validation files consisting of CAD models
from modelnet dataset.

To generate depth maps from these models run `generate_h5_data.sh` script which take three args,
output path to save h5 depth data, number of views for each model, output path to save raw depth data.

e.g., `sh generate_h5_data.sh /home/rash8327/Desktop/modelnet_off_h5 100 /home/rash8327/Desktop/modelnet_off_depth`