# Human Activity Recognition

## Overview

This project aims to build a Human Activity Recognition (HAR) system using the WISDM dataset. The system classifies activities such as walking, jogging, sitting, standing, accending and decending based on accelaretaion data. It includes data preprocessing, model training, and evaluation.

## Repository Structure

- `data/`: Contains the dataset and scripts for data loading and preprocessing(requirement running `setup.py`.
- `notebooks/`: Jupyter notebooks for exploratory data analysis, model development, and evaluation.
- `main/`: Saved models and scripts for training.
- `lib/`: Source code for data preprocessing, feature engineering, and model training.
- `scripts/`: Utility scripts for running experiments and generating reports.
- `results/`: Contains results from experiments, including plots and metrics.
- `REPORT.md`: Project documentation (more detail).
- `README.md`: Project documentation (this file).

## Getting Started

### Dataset

The WISDM dataset can be downloaded from the [WISDM website](http://www.cis.fordham.edu/wisdm/dataset.php).  
Note: you don't have to download dataset, `setup.py` can fetch and preprocess dataset.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rakawanegan/humanactivityrecognition_portfolio.git
   cd humanactivityrecognition_portfolio
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   python setup.py
   ```

### Running the Project

1. **Model Training**: 
   Train the machine learning model.
   ```bash
   python run.py
   ```

### Jupyter Notebooks

For detailed analysis and step-by-step implementation, refer to the Jupyter notebooks in the `notebooks/` directory. These notebooks cover data exploration, model development, and evaluation.

### Results

The results of the experiments, including accuracy, precision, recall, F1-score, and confusion matrices, can be found in the `results/` directory.
Note: Postprocess is here.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
