# Step 1: Use a lightweight Python base image
FROM python:3.8-slim

# Step 2: Set environment variables
ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    CONDA_ENV_NAME=orekit_venv_v12 \
    PYTHON_VERSION=3.8.20 \
    OREKIT_VERSION=12.1.2

# Step 3: Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y wget bzip2 unzip && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && apt-get clean && rm -rf /var/lib/apt/lists/*

# Step 4: Create Conda environment and install dependencies
RUN conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION && \
    conda install -n $CONDA_ENV_NAME -c conda-forge orekit=$OREKIT_VERSION matplotlib pandas numpy python-dateutil && \
    conda clean -afy

# Step 5: Set the working directory and copy requirements
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Install any additional requirements with pip inside the conda environment
RUN conda run -n $CONDA_ENV_NAME pip install --no-cache-dir -r /app/requirements.txt

# Step 6: Download the Orekit data inside the conda environment
RUN conda run -n $CONDA_ENV_NAME python -c "from orekit.pyhelpers import download_orekit_data_curdir; download_orekit_data_curdir()"


# JUST FOR THE PERSISTENCE BASELINE TO GET NECESSARY DATA
RUN conda run -n $CONDA_ENV_NAME python -c "from pymsis.utils import download_f107_ap; download_f107_ap()"
COPY orekit-data.zip /app/orekit-data.zip 


# Step 7: Copy source code
COPY submission.py atm.py propagator.py /app/
COPY ./trained_model/persistence_model.pkl /trained_model/persistence_model.pkl

# Ensure that the orekit-data is in the current directory
# After the download_orekit_data_curdir call, you should have /app/orekit-data/
# setup_orekit_curdir() in your code will pick it up from the current directory.

# Step 8: Set the default command
# Note: We run 'python submission.py' using conda run to ensure correct environment
CMD ["conda", "run", "-n", "orekit_venv_v12", "python", "submission.py"]


