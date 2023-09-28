# MLflow-project-template
MLflow project template

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```
conda create --prefix ./env python=3.7 -y
```

```
conda activate ./env
```
OR
```
source activate ./env
```

### STEP 04- install the requirements
```
pip install -r requirements.txt
```

### STEP 05- Create cond.yaml file - 
```
conda env export > conda.yaml
```

### STEP 06- commit and push the changes to the remote repository

...
## MLFlow commands

...
### Command to run MLproject file
```
mlflow run . --no-conda
```

...

### run any specific entry point in MLproject file
```
mlflow run . -e get_data --no-conda
```
...

### run any specific entry point in MLproject file
```
mlflow run . -e get_data -p config=configs/your_config.yaml --no-conda
```