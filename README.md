# DeepOCR - Text Detection and Recognition

## Overview

DeepOCR is an advanced text detection and recognition system that uses deep learning techniques to locate and recognize text in images and documents. This README provides instructions for setting up a Miniconda environment and installing the required dependencies to get DeepOCR up and running.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

## Setup Instructions

1. Clone the DeepOCR repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/DeepOCR.git
   cd DeepOCR
   ```

2. Create a Miniconda environment for DeepOCR (replace `deepocr_env` with your preferred environment name):

   ```bash
   conda env create -f conda/environment.yaml -n deepocr_env
   ```

   This command will create a new environment called `deepocr_env` and install all the required packages specified in the `conda/environment.yaml` file.

3. Activate the DeepOCR environment:

   ```bash
   conda activate deepocr_env
   ```

4. Install additional dependencies via pip:

   ```bash
   pip install -r requirements.txt
   ```

5. You may also need to install any additional GPU-specific packages if you are using GPU acceleration. Consult the documentation of your GPU for more details.

## Usage

You can now use DeepOCR within the activated Miniconda environment. You can run DeepOCR scripts, APIs, and web-based GUI interfaces as described in the project documentation.

Remember to activate the `deepocr_env` environment each time you want to use DeepOCR:

```bash
conda activate deepocr_env
```

run predictions on sample images

```bash
python3 pipeline.py
```

## Disclaimer

Please Note: The training scripts and data required to train DeepOCR models are not included in this repository due to licensing issues and data privacy considerations. 