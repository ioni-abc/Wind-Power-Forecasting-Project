import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from mlflow.models import infer_signature

if "MLFLOW_RUN_ID" in os.environ:
    del os.environ["MLFLOW_RUN_ID"]
if "MLFLOW_EXPERIMENT_ID" in os.environ:
    del os.environ["MLFLOW_EXPERIMENT_ID"]

def load_and_clean_data():

    print("- Loading and cleaning data -")
    
    # Read the files
    power_df = pd.read_csv('data/power.csv', parse_dates=["time"], index_col="time")
    wind_df = pd.read_csv('data/weather.csv', parse_dates=["time"], index_col="time")

    # Resample wind_df to 1 minute intervals to match power data
    wind_upsampled = wind_df.resample("1min").asfreq()

    # Interpolate numeric columns
    num_cols = ["Lead_hours", "Source_time", "Speed"]
    wind_upsampled[num_cols] = wind_upsampled[num_cols].interpolate()

    # Fill the NaN fields of `Direction`
    wind_upsampled["Direction"] = wind_upsampled["Direction"].ffill()

    # Join dataframes
    joined_dfs = power_df.join(wind_upsampled, how="inner")

    # Drop unnecessary columns
    cleaned_df = joined_dfs.drop(columns=["Lead_hours", "Source_time", "ANM", "Non-ANM"])

    # Filter Outliers (Stopped Turbines)
    # Logic: Speed > 5.0 m/s BUT Power < 1.0 MW
    fault_condition = (cleaned_df["Speed"] > 5.0) & (cleaned_df["Total"] < 1.0)
    cleaned_df = cleaned_df[~fault_condition]

    # Altering the wind direction (Sin/Cos Encoding)
    direction_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    cleaned_df["Degree"] = cleaned_df["Direction"].map(direction_map)
    cleaned_df["rad"] = np.deg2rad(cleaned_df["Degree"])
    cleaned_df["Sin_Dir"] = np.sin(cleaned_df["rad"])
    cleaned_df["Cos_Dir"] = np.cos(cleaned_df["rad"])

    # Handle NaNs
    cleaned_df = cleaned_df.dropna()
    
    return cleaned_df


def train_model(X_train, X_test, y_train, y_test, model_class, run_name, params=None, poly_degree=None):

    if params is None:
        params = {}

    with mlflow.start_run(run_name=run_name) as run:
    
        # A. Build the Pipeline Steps
        steps = [("Scaler", MinMaxScaler())]

        # Add Polynomial Features if requested (Crucial for the S-Curve fit)
        if poly_degree:
            steps.append(("Poly", PolynomialFeatures(degree=poly_degree)))
            mlflow.log_param("poly_degree", poly_degree)
    
        # Add the Model
        model_instance = model_class(**params)
        steps.append(("Model", model_instance))

        # B. Train
        print(f"\nTraining: {run_name}")
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        
        # C. Predict
        y_pred = pipeline.predict(X_test)
        
        # D. Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # E. Logging
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("model_type", str(model_class.__name__))
        
        # F. Save Model with Signature
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            pipeline, 
            "model", 
            signature=signature, 
            input_example=X_train.iloc[:2]
        )
        
        print(f"  MAE:  {mae:.4f} MW")
        print(f"  R2:   {r2:.4f}")
        print(f"  Run ID: {run.info.run_id}")


if __name__ == "__main__":
    
    # Setup MLflow Experiment
    mlflow.set_experiment("Wind_Power_Prediction")

    # Load and Clean Data
    df = load_and_clean_data()

    features = ["Speed", "Sin_Dir", "Cos_Dir"]
    X = df[features]
    y = df["Total"]

    # Split Data (Time Series Split - No Shuffle)
    # Train on Past (80%), Test on Future (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Experiment A: Baseline Linear Regression
    train_model(X_train, X_test, y_train, y_test, LinearRegression, "Linear_Baseline")

    # Experiment B: Ridge Regression (Alpha 1.0)
    train_model(X_train, X_test, y_train, y_test, Ridge, "Ridge_Alpha_1.0", params={"alpha": 1.0})

    # Experiment C: Polynomial Degree 2 (Parabola)
    train_model(X_train, X_test, y_train, y_test, LinearRegression, "Poly_Degree_2", poly_degree=2)

    # Experiment D: Polynomial Degree 3 (S-Curve)
    train_model(X_train, X_test, y_train, y_test, LinearRegression, "Poly_Degree_3", poly_degree=3)