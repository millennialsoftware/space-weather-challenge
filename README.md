# Space Weather Challenge

Welcome to the **Space Weather Challenge** repository. This project is part of the [STORM-AI](https://github.com/ARCLab-MIT/STORM-AI-devkit-2025) initiative, aiming to develop advanced AI algorithms for predicting space weather-driven changes in atmospheric density across low Earth orbit.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview

The objective of this project is to create an AI model that can nowcast and forecast atmospheric density variations caused by space weather phenomena. Accurate predictions are crucial for satellite tracking and orbit resilience modeling.

## Repository Structure

The repository is organized as follows:

- `Codabench_Submission`: Contains best model and associated files for Codabench Platform submission.
- `devkit/`: MIT Development toolkit with utilities and baseline models.
- `ml_pipeline/`: Directory for our training and processing modules
- `.gitignore`: Specifies files and directories to be ignored by Git.
baseline model.
- `data.dvc`: Data version control file for managing dataset.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- Docker (optional, for containerized environments)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/justin2213/space-weather-challenge.git
   cd space-weather-challenge
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   conda env create -f environment.yml
   conda activate myenv
   ```

## Usage

1. **Data Preparation:**

   Ensure that the necessary datasets are available and properly configured. Update the `data.dvc` file as needed.

2. **Configure Environment Variables**

   Create a `.env` file with variable PATH that points to where you are storing the dataset.

2.**Model Creation:**

   Create ml model as a class and store it as a python file inside the `ml_pipeline/models/` folder
   
3. **Model Training:**

   Train models using the `train_model.py` file in the `ml_pipeline/` directory or by running:

   ```bash
   cd ml_pipeline
   python train_model.py
   ```

This will train a specified model using the parameters that you set inside the train_model.py file. Once trained, the model will be tested against test and validation data and run through a submission generator which will calculate the propogation score compared to the previous best model. You can optionally save this model as the best model inside the `Codabench_Submission/` directory or inside of the `ml_pipeline/models/saved` directory. Each directory will contain the necessary files to submit to the Codabench platform. 
  
3. **Submission:**

   To generate the zip file that is necessary for Codabench submissions

   ```bash
   cd <model directory name>
   rm -f submission.zip && zip -r submission.zip . -x "submission.zip"
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


