# STORM-AI Starter Toolkit
## Persistence baseline
This baseline provide a boilerplate of how the submission should be made to the EvalAI platform. This notebook acts as a quick start guide and establishes a low performance baseline for the challenge. It consists in the replication of the inital values of the NRLMSIS model through all the output size.

As the objective is to get the orbital mean density, it utilizes the [devkit provided propagator](https://github.com/ARCLab-MIT/STORM-AI-devkit-2025/tree/main/orbit_propagator) to propagate from the initial state the orbit the object specified. 

> **Note:** The `Dockerfile` provides an image of the enviroment needed to run the propagator. However it is recommended to install it locally in a `cond` enviroment to be able to try it freely.

The structure is more or less the following
```
baselines
├── persistence                  
|    ├── propagator.py  # An adaptation of the propagator provided
|    ├── persistence-baseline.ipynb # Model creation and explanation
|    ├── orekit-data.zip # Used by the Orekit library
|    ├── atm.py  # Atmospheric model extrated from the notebook.
|    ├── Dockerfile # Used to build the image with the dependencies
|    └── requirements.txt # Depency list
└── evaluation.py # Script that calculates the challenge metrics
```
Once you run the notebook and the model is trained, you can follow the following steps to build and test your Docker submission:

- Build docker image for submission: 

```bash
docker build -t splid-submission .
```
- Test submission docker on a toy test dataset:

```bash
docker run -v [[TOY_TEST_DATASET_DIR]]:/dataset -v $(pwd)/submission:/submission splid-submission`
```
