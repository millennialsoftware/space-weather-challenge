# Troubleshooting

The following issues were encountered and resolved by members of the team during the development of this repo.
This may be of use for future users encountering the same issues.

## Problems during Training 

### Filename date stamps

> FileNotFoundError: [Errno 2] No such file or directory: '/code/space-weather-challenge/data/omni2/omni2-00000.csv'

Perhaps you have already downloaded the dataset, placed it in a directory, and configured the DATA_PATH environment variable to point to it.

The load_data.py functions are not expecting datestamps in the names, so e.g. `omni2-00000-20000603_to_20000802.csv` should be renamed to `omni2-00000.csv`. The bash script included in `ml_pipeline/preprocessing/truncate_date.sh` can be used to remove the datestamps.

### libdevice.10.bc

> libdevice not found at ./libdevice.10.bc

If you get this error, try adding a symlink to the libdevice.10.bc file that was installed
by conda. e.g. :

```bash
cd ml_pipeline
ln -s /home/your-user-name/.conda/envs/myenv/lib/libdevice.10.bc
python train_model.py
```

(reference https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path)

### NVIDIA Toolkit

You may need to install the NVIDIA toolkit. Consult https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### NVIDIA Libraries

You may need to install NVIDIA libraries into the conda environment. cuda-nvcc is the NVIDIA CUDA compiler driver. https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/

```bash
conda activate myenv
conda install -c nvidia cuda-nvcc
```

