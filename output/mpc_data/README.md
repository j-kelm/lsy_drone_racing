This folder should contain the `.hdf5` files with the raw data sampled from the MPC.

To view the `.hdf5` files, a tool like [myhdf5](https://myhdf5.hdfgroup.org/) is highly recommended.
The `.hdf5` files contain config groups with all configurations used for creating the datasets.
The configs are distributed in a hierarchical level to prevent duplication.

The entire process of generating training data from scratch is:
- Run `sample_trajs.py` to get an `.hdf5` file containing an initial trajectory as well as MPC and track config
- Run `data_pipeline.sh` which calls `sample_points.py` multiple times in parallel (this may take A LOT of time)
- Run `merge_hdfs.py` to merge all generated `.hdf5` files into one single file
- Run `compile_training_data.py` to generate an `.npz` file obtained from filtering and transforming the merged `.hdf5` file

The resulting `.npz` file can then be used to train a new diffusion policy with `train.py`.