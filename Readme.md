Run the main.py file to train the DDIM model on electricity dataset.

Please refer the config files from the config folder to go through or change the configs related to training and the diffusion process. 


# DDIM Model Training and Evaluation

## Model Training ('main.py')
This script is designed for training and evaluating a Denoising Diffuson Implicit Model (DDIM) for time-series generation. The script supports training a model from scratch.

#### Features
- Loads configuration settings from a YAML file.
- Supports training on different datasets (default: electricity dataset).
- Implements model training using Adam optimizer and multi-step learning rate scheduler.
- Saves configuration and model results for reproducibility in 'save' folder.

#### Requirements
The following dependencies are required to run the script:

- Python 3.x
- PyTorch
- YAML
- JSON
- argparse

#### Usage

##### Training a Model

To train a new model, run:
```bash
python main.py --config config.yaml --datatype electricity --device cuda:0 --seed 1
```


#### Arguments
- `--config`: Path to the configuration YAML file (default: `config.yaml`)
- `--datatype`: Dataset type (default: `electricity`)
- `--device`: Compute device, e.g., `cuda:0` or `cpu`
- `--seed`: Random seed for reproducibility (default: `1`)
- `--modelfolder`: Path to a saved model for evaluation
- `--nsample`: Number of samples for evaluation (default: `100`)

#### Output
- The script creates a directory under `./save/` for storing the trained model and configuration.
- Model weights are saved as `model.pth`.
- The configuration used is saved as `config.json`.

#### Notes
- Modify `config.yaml` to customize training hyperparameters.
- Ensure CUDA is available if training on GPU.


## Diffusion Process (`diffusion_process.py`)

### Overview

The `diffusion_process.py` script defines the `CSDI_Generation` class, which implements the forward diffusion. It handles noise scheduling and sample perturbation based on different diffusion schedules.

### Features

- Supports quadratic and linear diffusion schedules.
- Implements noise perturbation in the forward diffusion process.

### Key Components

- **`CSDI_Generation` class**: Initializes the diffusion process with model configurations.
- **`forward_diffusion(batch)`**: Applies noise perturbation to input data over a given number of diffusion steps.

## Model Architecture (`model.py`)

### Overview

The `model.py` script defines the core neural network architecture used in the CSDI model. It consists of several components, including transformers, residual blocks, and diffusion embeddings.

### Features

- Implements a transformer-based architecture for time-series generation.
- Uses a residual block structure for feature extraction.
- Includes diffusion embeddings for time-dependent representation.
- Supports both standard and linear attention mechanisms.

### Key Components

- **`DiffusionEmbedding` class**: Creates diffusion embeddings for time-dependent modeling.
- **`ResidualBlock` class**: Implements a residual connection with optional linear attention.
- **`diff_CSDI` class**: Defines the overall architecture, including input and output projections, residual layers, and diffusion embedding integration.



