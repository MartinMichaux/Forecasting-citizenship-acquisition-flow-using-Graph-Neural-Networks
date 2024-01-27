# Forecasting citizenship acquisition flow using Graph Neural Networks

### Getting started
After cloning the project, you need to perform the following steps to reproduce the results:
1. Create a virtual environment (```venv```) and setup Python interpreter, we have tested the code with Python 3.9.
2. Activate the virtual environment by running ```venv/Scripts/activate``` in a shell (on Windows).
3. Install dependencies in the venv by running ```pip install -r requirements.txt```

- Alternatively, the notebooks can be run in Google Colab as well. However, that requires manually adjusting some paths 
and also uploading the relevant files before processing.

### Repository Organization
- ```data```: contains the raw and processed data used in the project as well as the generated plots
- ```preprocessing```: includes the jupyter notebooks that were used to explore and preprocess the data for the models.
- ```models```: includes the jupyter nodebooks and scripts used for reproducing the baseline and graph based models' results.
