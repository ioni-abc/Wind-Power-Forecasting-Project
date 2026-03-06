### Which steps (preprocessing, retraining, evaluation) does the pipeline include?
- **Preprocessing:** Timestamp alignment via resampling or inner join, missing data handling, wind direction conversion to sine and cosine, scaling via `MinMaxScaler`, and feature engineering via `PolynomialFeatures`.
- **Retraining:** Fitting the defined scikit-learn pipeline on the past 80% of the dataset (`X_train`, `y_train`).
- **Evaluation:** Generating predictions on the future 20% of the dataset (`X_test`) and calculating MAE, RMSE, and R2 scores.

### What is the format of the data once it reaches the model?
- The data is a two-dimensional numerical array of floating-point numbers scaled between 0 and 1. 
- It contains the base features (Speed, Sin_Dir, Cos_Dir) and any generated polynomial interaction terms, depending on the specified degree.

### How did the data from the two sources get aligned?
- Timestamps from the MetOffice weather data and the wind power generation data were aligned using pandas datetime operations, typically through resampling to a common frequency or an inner join on the time index.

### How was the type of model and hyperparameters decided?
- The selection was based on the physical S-curve characteristics of wind turbine power generation. 
- Linear Regression and Ridge Regression (alpha=1.0) were selected to establish a linear baseline. 
- Polynomial degrees 2 and 3 were selected to test the model's ability to fit non-linear, cubic relationships.

### How is the newly trained model compared with the stored version?
- Comparison is done via the MLflow tracking dashboard. 
- The recorded evaluation metrics (MAE, RMSE, R2) of the new run are compared directly against the metrics logged in previous runs to determine if accuracy improved.

### How could the pipeline/system be improved?
- Implementing automated hyperparameter tuning (e.g., GridSearchCV).
- Utilizing `TimeSeriesSplit` for more robust cross-validation instead of a single 80/20 split.
- Adding automated data quality checks before the pipeline runs.

### How is it determined if the wind direction is a useful feature for the model?
- By training one model with the wind direction features (`Sin_Dir`, `Cos_Dir`) and a control model without them.
- If the evaluation metrics (R2 increases, MAE/RMSE decrease) improve on the test set, the feature is deemed useful. Feature importance coefficients can also be inspected.

### Regarding fetching the previous 90 days worth of data:

**How is it determined if this is a good interval?**
- By running experiments testing different historical intervals (e.g., 30, 60, 90, 180 days) and evaluating which dataset size produces the lowest MAE on a hold-out test set.

**What are the trade-offs when deciding on the interval?**
- Longer intervals provide more data to capture seasonal patterns but increase computation time and may include outdated operational trends.
- Shorter intervals train faster and reflect current turbine conditions but risk missing broader weather patterns.

**Would accuracy necessarily increase by including more data?**
- No. Older data might introduce noise if the turbine underwent maintenance, degraded, or if weather patterns shifted significantly, leading to worse predictions.


### Choice of models and evaluation metrics (What is logged in MLFlow and why?)
- **Models:** Linear, Ridge, and Polynomial regressions are used to test varying levels of complexity against the data.
- **Metrics Logged:** - MAE is logged to show the average prediction error in raw megawatts.
  - RMSE is logged to measure the penalty for large prediction errors.
  - R2 is logged to measure the proportion of variance explained by the model.

### How evaluation errors change in terms of cross-validation parameters
- Errors decreased as the polynomial degree parameter increased (MAE decreased from 4.92 MW to 3.61 MW).
- MLflow tracks different runs with different parameters (`alpha`, `poly_degree`), which isolates variable changes and identifies the exact configuration that yields the lowest error.

### Advantages of packaging experiments in MLFlow formats vs other options
- The `mlflow.sklearn.log_model` format automatically saves the model (`model.pkl`), the required environment dependencies (`conda.yaml`), and the expected data schema (`signature`).
- Compared to basic serialization options like a standard pickle file, MLflow prevents version mismatch errors and guarantees the environment can be exactly reproduced during deployment.


### Main reasons to deploy the model
- To provide actionable power generation forecasts for grid balancing and energy trading.
- To allow internal stakeholders to monitor wind farm efficiency against expected physical outputs.
- To make the predictive service available via an API for integration into broader automated energy management systems.