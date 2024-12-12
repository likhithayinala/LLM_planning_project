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
    git clone https://github.com/likhithayinala/GENAI_Project
    ```
2. **Navigate to the project directory**
    ```bash
    cd GENAI_Project
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training 
After installation, you can run train by first modifying the configuration file (in src/config/config.json) to include the dataset path. After which we can make additional modifications in scripts/train.sh to select model, token and layer for further specifications. Then run the following command.
```bash
bash scripts/train.sh
```

### Inference 
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
│   ├── dataset/
│   │   ├── judges/
│   │   └── dataset_generation/
│   ├── model_weights/
│   │   ├── refusal_model/
│   │   └── safety_model/
├── scripts/
│   ├── train.sh
    ├── inference.sh
│   ├── safety_judge_train.sh
├── requirements.txt
└── README.md
```

### `src/`

Contains the source code of the project.

- **`config/config.json`**: Configuration file specifying parameters for training.
- **`config/inf_config.json`**: Configuration file specifying parameters for inference.
- **`detection_model.py`**: Implements the detection model used in the project.
- **`training.py`**: Script to train the detection model.
- **`inference.py`**: Script to perform inference using the trained model.
- **`evaluate.py`**: Provides functions to evaluate the model's performance.
- **`dataloader.py`**: Handles data loading and preprocessing.
- **`dataset/`**: Contains code to generate complete datasets used in the project.
    - **`judges/`**: Contains python files to train judge models.
    - **`dataset_generation/`**: Contains python files to generate .h5 dataset files.
- **`model_weights/`**: Contains model weights to run inference.
    - **`safety_model/`**: Contains .pth model weights for safety.
    - **`refusal_model/`**: Contains .pth model weights for refusal.
Holds executable scripts for running training and inference.
### `scripts/`
- **`train.sh`**: Bash script to initiate model training using specified configurations.
- **`inference.sh`**: Bash script to perform inference with the trained model.
- **`safety_judge_train.sh`**: Bash script to train the safety judge model.

### `requirements.txt`

Lists the Python libraries and dependencies required to run the project.

### `README.md`

Provides an overview and instructions for the project.