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

    """
    I used pd.read_csv with parse_dates=["time"] and index_col="time" to load both the power and weather datasets. 
    parse_dates converts the time column from string to datetime and index_col makes time the index of the dataframe.
    """

    # print(wind_df)
    # print("\n--------------")
    # print(wind_df.resample("1min"))
    # print("\n--------------")
    # print(wind_df.resample("1min").asfreq())

    # Resample wind_df to 1 minute intervals to match power data
    wind_upsampled = wind_df.resample("1min").asfreq()

    """
    I used resample("1min") to upsample the wind data, adding a new row for every minute
    between the first time index and the last time index. Then I used asfreq() to tell pandas
    that for each new row it should put nan as value in the columns, except for the rows that already 
    had values.

    Before:
                                Direction  Lead_hours  Source_time     Speed
    time                                                                  
    2021-12-11 15:00:00+00:00       SSE           1   1639227600  11.17600
    2021-12-11 18:00:00+00:00       SSW           1   1639238400   8.04672
    2021-12-11 21:00:00+00:00       WSW           1   1639249200  11.17600
    2021-12-12 00:00:00+00:00       WSW           1   1639260000   8.94080
    2021-12-12 03:00:00+00:00        SW           1   1639270800   9.83488
    ...                             ...         ...          ...       ...
    2022-03-10 21:00:00+00:00         S           1   1646938800   9.83488
    2022-03-11 00:00:00+00:00       SSE           1   1646949600  11.17600
    2022-03-11 03:00:00+00:00       SSE           1   1646960400   9.83488
    2022-03-11 06:00:00+00:00       SSE           1   1646971200   8.94080
    2022-03-11 12:00:00+00:00        SE           1   1646992800  13.85824

    After:
                            Direction  Lead_hours   Source_time     Speed
    time                                                                   
    2021-12-11 15:00:00+00:00       SSE         1.0  1.639228e+09  11.17600
    2021-12-11 15:01:00+00:00       NaN         NaN           NaN       NaN
    2021-12-11 15:02:00+00:00       NaN         NaN           NaN       NaN
    2021-12-11 15:03:00+00:00       NaN         NaN           NaN       NaN
    2021-12-11 15:04:00+00:00       NaN         NaN           NaN       NaN
    ...                             ...         ...           ...       ...
    2022-03-11 11:56:00+00:00       NaN         NaN           NaN       NaN
    2022-03-11 11:57:00+00:00       NaN         NaN           NaN       NaN
    2022-03-11 11:58:00+00:00       NaN         NaN           NaN       NaN
    2022-03-11 11:59:00+00:00       NaN         NaN           NaN       NaN
    2022-03-11 12:00:00+00:00        SE         1.0  1.646993e+09  13.85824
    """

    # Interpolate numeric columns
    num_cols = ["Lead_hours", "Source_time", "Speed"]
    wind_upsampled[num_cols] = wind_upsampled[num_cols].interpolate()

    """
    I created the list num_cols to have only the columns that contain continuous numbers 
    ("Lead_hours", "Source_time", "Speed"). This is because the methodology used for filling the nan
    values for columns that contain numbers are different than columns that contain string (eg "Direction").
    I used the .interpolate() method to automatically estimate and fill in the missing NaN values for those 
    1-minute intervals.

    interpolate() uses linear interpolation. This means it draws a perfectly straight line between the 
    two known data points and calculates the exact mathematical value for every single minute in between

    Result:
                                Direction  Lead_hours   Source_time      Speed
    time                                                                    
    2021-12-11 15:00:00+00:00       SSE         1.0  1.639228e+09  11.176000
    2021-12-11 15:01:00+00:00       NaN         1.0  1.639228e+09  11.158615
    2021-12-11 15:02:00+00:00       NaN         1.0  1.639228e+09  11.141230
    2021-12-11 15:03:00+00:00       NaN         1.0  1.639228e+09  11.123845
    2021-12-11 15:04:00+00:00       NaN         1.0  1.639228e+09  11.106460
    ...                             ...         ...           ...        ...
    2022-03-11 11:56:00+00:00       NaN         1.0  1.646993e+09  13.803602
    2022-03-11 11:57:00+00:00       NaN         1.0  1.646993e+09  13.817261
    2022-03-11 11:58:00+00:00       NaN         1.0  1.646993e+09  13.830921
    2022-03-11 11:59:00+00:00       NaN         1.0  1.646993e+09  13.844580
    2022-03-11 12:00:00+00:00        SE         1.0  1.646993e+09  13.858240
    """

    # Fill the NaN fields of `Direction`
    wind_upsampled["Direction"] = wind_upsampled["Direction"].ffill()

    """
    I used .ffill() (forward fill) on the Direction column to handle the missing categorical data.
    I used this method because it takes the last known valid observation and copies it forward to replace 
    all the NaN values until it hits the next valid forecast.

    Result:
                                Direction  Lead_hours   Source_time      Speed
    time                                                                    
    2021-12-11 15:00:00+00:00       SSE         1.0  1.639228e+09  11.176000
    2021-12-11 15:01:00+00:00       SSE         1.0  1.639228e+09  11.158615
    2021-12-11 15:02:00+00:00       SSE         1.0  1.639228e+09  11.141230
    2021-12-11 15:03:00+00:00       SSE         1.0  1.639228e+09  11.123845
    2021-12-11 15:04:00+00:00       SSE         1.0  1.639228e+09  11.106460
    ...                             ...         ...           ...        ...
    2022-03-11 11:56:00+00:00       SSE         1.0  1.646993e+09  13.803602
    2022-03-11 11:57:00+00:00       SSE         1.0  1.646993e+09  13.817261
    2022-03-11 11:58:00+00:00       SSE         1.0  1.646993e+09  13.830921
    2022-03-11 11:59:00+00:00       SSE         1.0  1.646993e+09  13.844580
    2022-03-11 12:00:00+00:00        SE         1.0  1.646993e+09  13.858240
    """

    # Join dataframes
    joined_dfs = power_df.join(wind_upsampled, how="inner")

    # Drop unnecessary columns
    cleaned_df = joined_dfs.drop(columns=["Lead_hours", "Source_time", "ANM", "Non-ANM"])

    """
    - I used the .join(how="inner") method to merge the power data with the new upsampled weather forecasts.
    - I used an "inner" join to merge the datasets, ensuring that the final dataset only keeps rows where the 
    timestamps exist in both dataframes.
    - Because index_col="time", time is the column where the join occurs.
    - I used the .drop(columns=[...]) method to remove features that are irrelevant to predicting power output.
    - I eliminated "ANM" and "Non-ANM" (which the assignment description states are not relevant), 
    and "Lead_hours" and "Source_time". They are not actual physical weather conditions (like wind speed) 
    that dictate how fast a turbine spins.

    Result:
                                Total Direction      Speed
    time                                                     
    2021-12-11 15:00:00+00:00  27.864560       SSE  11.176000
    2021-12-11 15:01:00+00:00  28.489561       SSE  11.158615
    2021-12-11 15:02:00+00:00  28.150561       SSE  11.141230
    2021-12-11 15:03:00+00:00  26.634557       SSE  11.123845
    2021-12-11 15:04:00+00:00  26.172559       SSE  11.106460
    ...                              ...       ...        ...
    2022-03-11 11:56:00+00:00  32.207000       SSE  13.803602
    2022-03-11 11:57:00+00:00  32.628001       SSE  13.817261
    2022-03-11 11:58:00+00:00  32.735001       SSE  13.830921
    2022-03-11 11:59:00+00:00  32.435002       SSE  13.844580
    2022-03-11 12:00:00+00:00  32.428001        SE  13.858240
    """

    # Filter Outliers (Stopped Turbines)
    # Logic: Speed > 5.0 m/s BUT Power < 1.0 MW
    fault_condition = (cleaned_df["Speed"] > 5.0) & (cleaned_df["Total"] < 1.0)
    cleaned_df = cleaned_df[~fault_condition]

    """
    - I used a boolean mask called fault_condition to identify data points where the wind speed was 
    high (greater than 5.0 m/s) but the total power generation was near zero (less than 1.0 MW).
    - I used the AND operator (&) to ensure both of these conditions had to be met simultaneously 
    for a row to be flagged.
    - I used the tilde operator (~) to invert that mask when filtering the dataframe 
    (cleaned_df[~fault_condition]), effectively saying "keep everything that is NOT a fault."

    Fault condition looks like:

    2021-12-11 15:00:00+00:00    False
    2021-12-11 15:01:00+00:00    False
    2021-12-11 15:02:00+00:00    False
    2021-12-11 15:03:00+00:00    False
    2021-12-11 15:04:00+00:00    False
                                ...  
    2022-03-11 11:56:00+00:00    False
    2022-03-11 11:57:00+00:00    False
    2022-03-11 11:58:00+00:00    False
    2022-03-11 11:59:00+00:00    False
    2022-03-11 12:00:00+00:00    False
    """

    print(cleaned_df["Direction"].unique())
    # ['SSE' 'SSW' 'WSW' 'SW' 'S' 'SE' 'W' 'WNW' 'NW' 'NNW' 'NNE' 'NE' 'ENE' 'E' 'ESE' 'N']

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

    """
    - I used a dictionary called direction_map to translate the 16 categorical compass strings (like 'N' or 'SW') 
    into their corresponding numerical degrees on a 360-degree circle.
    - I used the .map() function to apply this dictionary to the entire "Direction" column, creating a brand new numerical 
    "Degree" column.
    - I used np.deg2rad() to convert those degrees into radians, because Python's mathematical functions expect 
    angles to be in radians rather than degrees.
    - I used np.sin() and np.cos() from the NumPy library to transform those radians into two separate 
    features: "Sin_Dir" and "Cos_Dir".

    Creating the sin/cos columns is crucial for our model training later. The model would interprete 1 degree
    and 359 degrees as something vastly different, in the same way that in a ruler 1 is very far from 359. But
    in terms of direction, these two numbers are practicaly the same thing. So by using sin/cos we tell the model
    to think in terms of a circle, where sin/cos act as coordinates.

    With degree:
                                Total Direction      Speed  Degree
    time                                                             
    2021-12-11 15:00:00+00:00  27.864560       SSE  11.176000   157.5
    2021-12-11 15:01:00+00:00  28.489561       SSE  11.158615   157.5
    2021-12-11 15:02:00+00:00  28.150561       SSE  11.141230   157.5
    2021-12-11 15:03:00+00:00  26.634557       SSE  11.123845   157.5
    2021-12-11 15:04:00+00:00  26.172559       SSE  11.106460   157.5
    ...                              ...       ...        ...     ...
    2022-03-11 11:56:00+00:00  32.207000       SSE  13.803602   157.5
    2022-03-11 11:57:00+00:00  32.628001       SSE  13.817261   157.5
    2022-03-11 11:58:00+00:00  32.735001       SSE  13.830921   157.5
    2022-03-11 11:59:00+00:00  32.435002       SSE  13.844580   157.5
    2022-03-11 12:00:00+00:00  32.428001        SE  13.858240   135.0

    [108842 rows x 4 columns]

    ---------------------------------------------------------------------------

    With rad:
                                Total Direction      Speed  Degree       rad
    time                                                                       
    2021-12-11 15:00:00+00:00  27.864560       SSE  11.176000   157.5  2.748894
    2021-12-11 15:01:00+00:00  28.489561       SSE  11.158615   157.5  2.748894
    2021-12-11 15:02:00+00:00  28.150561       SSE  11.141230   157.5  2.748894
    2021-12-11 15:03:00+00:00  26.634557       SSE  11.123845   157.5  2.748894
    2021-12-11 15:04:00+00:00  26.172559       SSE  11.106460   157.5  2.748894
    ...                              ...       ...        ...     ...       ...
    2022-03-11 11:56:00+00:00  32.207000       SSE  13.803602   157.5  2.748894
    2022-03-11 11:57:00+00:00  32.628001       SSE  13.817261   157.5  2.748894
    2022-03-11 11:58:00+00:00  32.735001       SSE  13.830921   157.5  2.748894
    2022-03-11 11:59:00+00:00  32.435002       SSE  13.844580   157.5  2.748894
    2022-03-11 12:00:00+00:00  32.428001        SE  13.858240   135.0  2.356194

    [108842 rows x 5 columns]

    ---------------------------------------------------------------------------

    With sin cos:
                                Total Direction      Speed  Degree       rad   Sin_Dir   Cos_Dir
    time                                                                                           
    2021-12-11 15:00:00+00:00  27.864560       SSE  11.176000   157.5  2.748894  0.382683 -0.923880
    2021-12-11 15:01:00+00:00  28.489561       SSE  11.158615   157.5  2.748894  0.382683 -0.923880
    2021-12-11 15:02:00+00:00  28.150561       SSE  11.141230   157.5  2.748894  0.382683 -0.923880
    2021-12-11 15:03:00+00:00  26.634557       SSE  11.123845   157.5  2.748894  0.382683 -0.923880
    2021-12-11 15:04:00+00:00  26.172559       SSE  11.106460   157.5  2.748894  0.382683 -0.923880
    ...                              ...       ...        ...     ...       ...       ...       ...
    2022-03-11 11:56:00+00:00  32.207000       SSE  13.803602   157.5  2.748894  0.382683 -0.923880
    2022-03-11 11:57:00+00:00  32.628001       SSE  13.817261   157.5  2.748894  0.382683 -0.923880
    2022-03-11 11:58:00+00:00  32.735001       SSE  13.830921   157.5  2.748894  0.382683 -0.923880
    2022-03-11 11:59:00+00:00  32.435002       SSE  13.844580   157.5  2.748894  0.382683 -0.923880
    2022-03-11 12:00:00+00:00  32.428001        SE  13.858240   135.0  2.356194  0.707107 -0.707107
    """

    # Handle NaNs
    cleaned_df = cleaned_df.dropna()

    """
    - I used .dropna() at the very end as a final safety step, in case something went wrong in the previous
    steps.
    """
    
    return cleaned_df


def train_model(X_train, X_test, y_train, y_test, model_class, run_name, params=None, poly_degree=None):

    if params is None:
        params = {}

    with mlflow.start_run(run_name=run_name) as run:
        """
        Starts a recording session. Every setting and result inside this block is saved to the MLflow dashboard. 
        """
    
        # A. Build the Pipeline Steps
        steps = [("Scaler", MinMaxScaler())]
        """
        MinMaxScaler squishes every feature into a range between 0 and 1.
        """

        # Add Polynomial Features if requested (Crucial for the S-Curve fit)
        # This is feature engineering
        if poly_degree:
            steps.append(("Poly", PolynomialFeatures(degree=poly_degree)))
            mlflow.log_param("poly_degree", poly_degree)
            """
            Polynomial fits better than Linear Regression because wind turbine power generation 
            isn't a straight line but an S-shaped curve.
            """
    
        # Add the Model
        model_instance = model_class(**params)
        """
        I used the ** unpacking operator to take the params dictionary and inject its contents 
        into the model_class. This unpacks a dictionary like {"alpha": 1.0} and turns 
        it into actual code: Ridge(alpha=1.0). Params is present only for the Ridge model.
        """
        steps.append(("Model", model_instance))
        """
        model_instance is model_class which either LinearRegression or Ridge, and has params
        only in the Ridge.

        There are two or three steps in the pipeline: MinMaxScaler, PolynomialFeatures if poly_degree, model_instance.
        """

        # B. Train
        print(f"\nTraining: {run_name}") # run_name is one of these: Linear_Baseline, Ridge_Alpha_1.0, Poly_Degree_2, Poly_Degree_3
        pipeline = Pipeline(steps)
        """
        I used the Scikit-Learn Pipeline class to bundle the scaling (minmaxscaler), 
        feature engineering (PolynomialFeatures), and modeling (model_instance) steps together.
        """
        pipeline.fit(X_train, y_train)
        """
        I used the .fit() method, passing in X_train and y_train, to execute the training process. 
        This pushes the raw training data through the scaling and transformation steps before finally 
        training the model to predict the true power generation targets
        """
        
        # C. Predict
        y_pred = pipeline.predict(X_test)
        """
        I used the .predict() method on the trained pipeline object, passing in the unseen X_test dataset 
        (the future weather conditions), to generate the model's estimated power generation values, 
        which I stored in the y_pred variable.
        """
        
        # D. Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        """
        - I used the mean_absolute_error function to calculate the MAE, which provides a straightforward average 
        of exactly how far off my model's predictions were in megawatts (MW).
        - I used the mean_squared_error function, wrapped in NumPy's square root function (np.sqrt()), 
        to calculate the RMSE. This specific metric heavily penalizes the model for making exceptionally large, 
        singular prediction errors.
        - I used the r2_score function to calculate the R-squared value, which acts as a percentage grade 
        indicating how much of the actual variance in power generation my model successfully explained.
        """
        
        # E. Logging
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("model_type", str(model_class.__name__))
        """
        - I used the mlflow.log_params() function to record the exact dictionary of settings 
        (like the Ridge alpha penalty) used for the current model.
        - I used the mlflow.log_metric() function three times to save my calculated MAE, RMSE, and R-squared scores 
        to the tracking server.
        - I used the mlflow.set_tag() function to attach a label (e.g., "LinearRegression" or "Ridge") to the experiment run.
        """
        
        # F. Save Model with Signature
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            pipeline, 
            "model", 
            signature=signature, 
            input_example=X_train.iloc[:1]
        )
        """
        - I used infer_signature to create a blueprint of the data. It records the names and types of the columns 
        so the model knows exactly what kind of data to expect in the future.
        - I used mlflow.sklearn.log_model to save the trained pipeline. This stores the model permanently 
        so it can be reused later without having to retrain it from scratch.
        - I used the input_example parameter to save a single row of real data alongside the model, 
        providing a clear example of what the inputs should look like.
        """
        
        print(f"  MAE:  {mae:.4f} MW")
        print(f"  R2:   {r2:.4f}")
        print(f"  Run ID: {run.info.run_id}")


if __name__ == "__main__":
    
    # Setup MLflow Experiment
    mlflow.set_experiment("Wind_Power_Prediction")

    # Load and Clean Data
    df = load_and_clean_data()

    # This is feature selection
    features = ["Speed", "Sin_Dir", "Cos_Dir"]
    X = df[features]
    # Target variable. It is the total power generation.
    y = df["Total"]

    # Split Data (Time Series Split - No Shuffle)
    # Train on Past (80%), Test on Future (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Experiment A: Baseline Linear Regression
    train_model(X_train, X_test, y_train, y_test, LinearRegression, "Linear_Baseline")
    """
    The simplest possible model. It tries to fit a straight line to the data.
    """

    # Experiment B: Ridge Regression (Alpha 1.0)
    train_model(X_train, X_test, y_train, y_test, Ridge, "Ridge_Alpha_1.0", params={"alpha": 1.0})
    """
    Ridge is like Linear Regression, but keeps your model from jumping to extreme conclusions, and alpha 
    is telling the code exactly how strict to be about it.
    """

    # Experiment C: Polynomial Degree 2 (Parabola)
    train_model(X_train, X_test, y_train, y_test, LinearRegression, "Poly_Degree_2", poly_degree=2)
    """
    Creates a parabolic fit. It can curve once, which is better than a straight line but still 
    doesn't quite capture the S-curve.
    """

    # Experiment D: Polynomial Degree 3 (S-Curve)
    train_model(X_train, X_test, y_train, y_test, LinearRegression, "Poly_Degree_3", poly_degree=3)
    """
    Creates a cubic fit. This is the sweet spot for wind turbines because a cubic equation can 
    create the S-shape required to model power generation.
    """