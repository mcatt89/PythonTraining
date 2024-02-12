Notes on setting up the python environment in Windows

I am using Visual Studio Code with PowerShell and Python extensions from the marketplace

Install python for Windows from: https://www.python.org/downloads/
I'm running 3.12 at the time of writing

modules/libraries
use pip to install modules in python, it is included in python after version 3.4
commands below are from Win11, run in PowerShell Administrative terminal

upgrade pip as needed
py -m pip install --upgrade pip

numpy:
py -m pip install numpy

pandas:
py -m pip install pandas

seaborn:
py -m pip install seaborn

matplotlib:
py -m pip install matplotlib

scikit-learn: 
py -m pip install -U scikit-learn
NOTE: after install that is a capitol U 
