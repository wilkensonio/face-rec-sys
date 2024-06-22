# CSC-581 Team Project  
Contributors: Wilkenson | Aeron | Kyle 

# Face Recognition Classifier 
This Python script performs facial recognition using various machine learning models on a dataset of facial landmarks (`.pts` files). It extracts features from the landmarks, organizes the data into training and testing sets, trains several classifiers, evaluates their performance, and generates classification reports along with confusion matrices.

## Overview
The script performs the following tasks:
- Loads facial landmark data from `.pts` files in specified directories.
- Extracts and filters data points.
- Computes features from the facial landmarks.
- Organizes data into training and testing sets.
- Trains and evaluates multiple machine learning models:
  - Artificial Neural Network (ANN)
  - k-Nearest Neighbors (kNN)
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Decision Tree
- Generates classification reports and confusion matrices for each model.
- Saves reports and plots in designated folders.
folders(classification_reports, confusion_matrices_plt)

## Requirements
If there are multiple python -v projects consider activating a
virtual env.

cd in the parent folder CSC481A
```bash
python -m venv /path/to/new/virtual/environment
```
| Platform   | Shell              | Command to activate virtual environment                  |
|------------|--------------------|----------------------------------------------------------|
| POSIX      | bash/zsh           | `$ source <venv>/bin/activate`                           |
| POSIX      | fish               | `$ source <venv>/bin/activate.fish`                      |
| POSIX      | csh/tcsh           | `$ source <venv>/bin/activate.csh`                       |
| PowerShell | PowerShell (Windows)| `PS C:\> <venv>\Scripts\Activate.ps1`                    |
| Windows    | cmd.exe            | `C:\> <venv>\Scripts\activate.bat`                       |
| PowerShell | PowerShell (Windows)| `PS C:\> <venv>\Scripts\Activate.ps1`                    |

 
- Python 3.x
- Required Python packages: `numpy`, `matplotlib`, `scikit-learn`

Install required packages using pip or pip3:

```bash
pip3 install numpy matplotlib scikit-learn  
```
or
cd in the parent folder CSC481A

prefered
```bash
pip3 install requirement.txt
```
# Run 
python3 or python

cd into parent folder and run script
```bash
python3 main.py
```
must be in the same folder as the main.py module

once the script is done executing message will display in the terminal 
then the report and plot will be save in thir respective folders.






