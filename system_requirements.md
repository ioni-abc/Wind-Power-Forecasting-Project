### 1. Reads the latest data
- Implemented in the `load_and_clean_data` function.
- MetOffice weather data and wind power generation data are loaded into pandas DataFrames.

### 2. Prepares the data for model training
- **Aligning timestamps & Handling missing data:** Executed in the `load_and_clean_data` function prior to training.
- **Altering the wind direction to be a usable feature:** Compass directions are mapped to mathematical representations using the `Sin_Dir` and `Cos_Dir` features.
- **Scaling the data to be within a set range:** Executed using `MinMaxScaler()` within the Scikit-Learn Pipeline to scale features to a range between 0 and 1.

### 3. Trains a (few) regression models & tracks with MLflow
- **Models trained:** Linear Regression (Baseline), Ridge Regression (Alpha 1.0), Polynomial Regression (Degree 2), and Polynomial Regression (Degree 3).
- **Tracking:** Implemented using the `with mlflow.start_run(...)` context manager. Performance is evaluated and tracked using MAE, RMSE, and R2 metrics.

### 4. Saves the best performing model to disk with parameters and metrics
- **Logging parameters and metrics:** Executed using `mlflow.log_params()` and `mlflow.log_metric()`.
- **Saving to disk:** The trained model pipeline is saved to disk using `mlflow.sklearn.log_model(pipeline, "model", signature=signature, input_example=...)`.
