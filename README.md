# helix4262
ModNet: Developed by Lim Hui Xuan, Lim Tze Xin, Ong Yi Lin and Wang Yiling

## Table of Contents
- Folder Structure (TBC)
- Software Requirements 
- Setting Up (TBC)
    - Machine Setup
    - Cloning Repository
    - Install Software & Packages
- Running Scripts (TBC)
    - Training a New Model
    - Predicting using Model

## Folder Structure (TBC)

## Software Requirements

1. Python version >= 3.8

## Setting Up

### Machine Setup  (to verify)

1. AWS Instance: If you are running on the AWS Instance, start a new instance that is at least t3.large. This is to ensure that the model does not run too long. Note that the default Python version for the instance will be at least Python 3.8.

2. Locally : If you are running locally, ensure that you have at least Python version ≥ 3.8

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

## Running Scripts

Change directory into the `code` folder to run the scripts for training and predicting.

```bash
cd code
```

After running the above scripts, you may proceed here to learn how to train new models and predict using existing models.

