## 2. Training Level

The training level includes the code that processes data and trains the machine learning model.

### Training Structure

Training code is organized in the `training/` directory:

```
training/
├── data_models/                                    # Data model definitions
│   ├── __init__.py
│   └── media_data_frame.py                         # Polars DataFrame wrapper class
├── training/                                       # Training pipeline components
│   ├── __init__.py
│   ├── _00_error_correction.py                     # Ad hoc error correction
│   ├── _01_extract_media_table.py                  # Data extraction
│   ├── _02_xgb_binomial_classifier_prep_data.py    # Data preparation
│   └── _03_xgb_binomial_classifier_build_model.py  # Model training
```

### Modeling Approach

#### Problem Formulation

For the initial version of this model I am going with a binomial classifier, largely because that is the type of model that will best fit my current model training data. As an output of the services deployed via my [automatic-transmission](https://github.com/x81k25/automatic-transmission) repository, I have multiple status flags attached to several thousand media records that can be used as a true/false label for media selection.

It would be interesting to potentially try a multi-class classification problem in the future. When discussing with my wife, we have considered labeling the data as one of these possibilities: `["would-not-watch", "would watch", "would watch multiple times"]` or something analogous. This would likely decrease the probability of getting false negatives on movies we would enjoy the most by giving them their own distinct class.

#### Algorithm Selection

The primary model uses XGBoost for binary classification, as it handles complex feature interactions well and provides excellent performance with tabular data.

### Training Pipeline

The training pipeline follows these steps:

1. **Data Extraction** (`_01_extract_media_table.py`): 
   - Extracts media data from PostgreSQL database
   - Exports raw data to Parquet format

2. **Data Preparation** (`_02_xgb_binomial_classifier_prep_data.py`):
   - Filters for movie data
   - Creates labels from rejection status
   - Normalizes numeric features
   - Encodes categorical variables (genres, languages)
   - Exports training data to Parquet format

3. **Model Training** (`_03_xgb_binomial_classifier_build_model.py`):
   - Trains an XGBoost binary classifier
   - Uses grid search for hyperparameter optimization
   - Evaluates model performance (accuracy, precision, recall, F1, AUC)
   - Logs model metrics and parameters to MLflow
   - Exports prediction results and model artifacts

### Running Training Components

Run individual components of the training pipeline:

```python
# Extract data
from training.core import extract_media
extract_media()

# Prepare data
from training.core import xgb_prep
xgb_prep()

# Train model
from training.core import xgb_build
xgb_build()
```

### MLflow Tracking

This project uses MLflow to track model experiments. For information on setting up and using MLflow, refer to the [official MLflow documentation](https://mlflow.org/docs/latest/index.html).

When training the model, metrics are automatically logged to MLflow, including:
- Accuracy, precision, recall, F1 score
- ROC-AUC
- Confusion matrix components (TP, FP, TN, FN)
- Model hyperparameters