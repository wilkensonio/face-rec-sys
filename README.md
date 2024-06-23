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

 
# Run 

`step-1 `

Navigate to the parent directory of the project and type one of the commands below 

## MacOS: `bash run`

## Windows: `bash runw` 
**If running one of the command above raise and error go to [step-2](#step-2-requirements).**

To see your current location type the command pwd in the terminal.

If you see this message 
```Report is being generated ...

✅ Report generated successfully 🗂️ 🗂️

1. Check the classification_reports folder for the report

2. Check the roc_curve_plt folder for the ROC curve plots

3. Check the confusion_matrices_plt folder for the confusion matrices plots

The folders can be found in the root directory of the project
``` 
in the terminal the execution was succesfull. 

4 new folders have been created:
1. classification_reports
2. confusion_matrices_plt
3. roc_curves_plt
4. roc_repot

this folders contain all the reports and plots. 
 
---
<a name="step-2-requirements"></a>`Step 2` 

requirements

- Python 3.x
- Required Python packages: In requirement.txt file 

Install required packages using pip or pip3:
 
cd in the parent folder CSC481A  
```bash
pip3 install -r requirement.txt
```
cd into parent folder and run script

In the terminal type `python or python3 main.py` and enter, if it runs successfully read the message in the terminal  
 
must be in the same folder as the main.py module if not the command will not work

once the script is done executing message will display in the terminal 
then the report and plot will be save in their respective folders.

4 new folders have been created:
1. classification_reports
2. confusion_matrices_plt
3. roc_curves_plt
4. roc_repot
 
```Report is being generated ...

✅ Report generated successfully 🗂️ 🗂️

1. Check the classification_reports folder for the report

2. Check the roc_curve_plt folder for the ROC curve plots

3. Check the confusion_matrices_plt folder for the confusion matrices plots

The folders can be found in the root directory of the project
```





