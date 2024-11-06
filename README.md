# Predicting m6A methylation sites using neural networks
## Team helix4262
ModNet: Developed by Lim Hui Xuan, Lim Tze Xin, Ong Yi Lin and Wang Yiling

## Project Description

This project aims to leverage on computational methods to develop and train a machine learning model capable of detecting and predicting m6A methylation sites across various datasets from different cancer cell lines. By improving the accuracy and efficiency of m6A site detection, this model can potentially enhance early cancer detection, ultimately contributing to a reduction in cancer-related mortality. The final model is ModNet with neural network architecture.

## Citation

If you use this software in your research, please cite it as follows:

Lim, H. X., Lim, T. X., Ong, Y. L., & Wang, Y. (2024). *Predicting m6A modified sites on RNA data with ModNet* (Version 1.0.0) [Computer software]. GitHub. https://github.com/OYL02/helix4262. https://doi.org/10.5281/zenodo.14046279

## Table of Contents
- Folder Structure
- Software Requirements 
- Setting Up 
    - Machine Setup
    - Cloning Repository
    - Install Software & Packages

## Folder Structure

```
.
├── code/
├── dataset/
├── diagrams/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Software Requirements

1. Python version >= 3.8

## Setting Up

### Machine Setup  

1. AWS Instance: If you are running on the AWS Instance, start a new instance that is at least t3.large. This is to ensure that the model does not run too long. Note that the default Python version for the instance will be at least Python 3.8.

2. Locally : If you are running locally, ensure that you have at least Python version ≥ 3.8
3. You may check the Python version with the following command:
```
python3 --version
```

### Cloning Repository

To clone the repository into your virtual machine or local device, run the following command:
```
git clone https://github.com/OYL02/helix4262.git
```

### Installing Software and Packages

After the repository has been cloned, run the following commands to install the relevant software or packages:

1. Installing `pip` (if you are running on an AWS instance):

```bash
sudo apt install python3-pip -y
```

2. Create a virtual environment (if you are running on a local instance):

```bash
python -m venv rna_predictions # creates the virtual environment 'rna_predictions'
./rna_predictions/Scripts/Activate # to run the virtual environment
```

3. Installing packages required

Before installing the packages, ensure that you are within the `helix4262` folder. You can do so by using the following command:

```bash
cd helix4262
```

Now, proceed to install all required packages as indicated in the text file `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

These packages will be included in your virtual environment and the virtual environment will have to be activated before running any of the scripts in this repository.


After running the above commands, you may proceed [here](https://github.com/OYL02/helix4262/tree/main/code) to learn how to train new models and predict using existing models.

