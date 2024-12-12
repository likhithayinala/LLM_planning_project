# EECSE6694_Project
## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [CodeStructure](#codestructure)
## Introduction

EECSE6694_Project is a comprehensive project developed as part of the EECSE6694 course. It aims to provide solutions and insights into how LLMs work and predict future tokens. We use these findings to early stop an LLM in case the answer doesn't align to the given specifications. In our project specifically, this task is for preventing LLM refusal and unsafe answers.

## Installation

To set up the project locally, please follow these steps:

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/EECSE6694_Project.git
    ```
2. **Navigate to the project directory**
    ```bash
    cd EECSE6694_Project
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

# Training 
After installation, you can run train by first modifying the configuration file (in src/config/config.json) to include the dataset path. After which we can make additional modifications in scripts/train.sh to select model, token and layer for further specifications. Then run the following command.
```bash
bash scripts/train.sh
```
For detailed usage instructions and examples, refer to the [user guide](docs/USER_GUIDE.md).

# Inference 
If you want to test the proposed pipeline, you need access to Llama 2 7B in huggingface. If you have access, you can input your credentials using:
```bash
huggingface-cli login
```
Once logged in, you have to run the below command where you'll be asked to input your prompt
```bash
bash scripts/inference.sh
```

## Code Structure

The project is organized into the following directories and files:

```
EECSE6694_Project/
├── src/
│   ├── config/
│   │   └── config.json
│   │   └── inf_config.json
│   ├── training.py
|   ├── inference.py
│   ├── dataloader.py
│   ├── dataset/es
│   │   ├── judges/
│   │   └── dataset_generation/
│   ├── train.sh
    ├── safety_judge_train.sh
│   ├── safety_judge_train.sh
├── requirements.txt
└── README.md
```

### `src/`

Contains the source code of the project.

- **`config/config.json`**: Configuration file specifying parameters for training and inference.
- **`config/config.json`**: Configuration file specifying parameters for training.
- **`config/inf_config.json`**: Configuration file specifying parameters for inference.- **`modules/model.py`**: Defines the neural network architecture used for language modeling.
- **`detection_model.py`**: Implements the detection model used in the project.
- **`training.py`**: Script to train the detection model.- **`modules/training.py`**: Includes functions to train the model with given datasets and configurations.
- **`inference.py`**: Script to perform inference using the trained model.
- **`evaluate.py`**: Provides functions to evaluate the model's performance.- **`/evaluation.py`**: Provides functions to evaluate the model's performance on validation and test sets.
- **`dataloader.py`**: Handles data loading and preprocessing.
- **`dataset/`**: Contains datasets used in the project.
    - **`judges/`**: Contains datasets related to judge models.### `scripts/`
    - **`dataset_generation/`**: Scripts or data for generating datasets.
Holds executable scripts for running training and inference.
### `scripts/`
- **`train.sh`**: Bash script to initiate model training using specified configurations.
Contains executable scripts for running training and inference.
- **`inference.sh`**: Bash script to perform inference with the trained model.
- **`train.sh`**: Bash script to initiate model training.
- **`safety_judge_train.sh`**: Bash script to train the safety judge model.### `requirements.txt`
- **`inference.sh`**: Bash script to perform inference.
Lists the Python libraries and dependencies required to run the project.
### `requirements.txt`

Lists the Python libraries and dependencies required to run the project.

### `README.md`

Provides an overview and instructions for the project.