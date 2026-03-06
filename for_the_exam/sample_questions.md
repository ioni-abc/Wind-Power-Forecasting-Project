### Question 1: Time-series Data Splitting Strategy
**Question:** The training process uses a specific splitting method. Describe the method used, why it is necessary for wind forecasting, and the consequences of using a standard randomized split.

**Answer:**
The data is split using `train_test_split` with `shuffle=False`. This ensures the model trains on the first 80% of the data (the past) and is evaluated on the final 20% (the future).

Chronological order is vital for time-series forecasting. If the data were shuffled, the model would use future data to predict past events, leading to "data leakage." This creates artificially high accuracy scores that would fail in real-world production where the future is unknown.


```python
# Splitting the data chronologically
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
Question 2: Feature Selection vs. Feature EngineeringQuestion: Distinguish between feature selection and feature engineering in the context of the wind forecasting script. Provide the code that executes both.Answer:Feature selection is the process of choosing which raw columns to use as inputs. Feature engineering is the creation of new mathematical features to help the model learn complex patterns.In this project, feature selection isolates "Speed" and the direction coordinates. Feature engineering occurs inside the pipeline using PolynomialFeatures, which creates squared ($x^2$) and cubed ($x^3$) versions of the wind speed to capture the S-shaped physical power curve of the turbine.Python# Feature selection: Picking raw columns
features = ["Speed", "Sin_Dir", "Cos_Dir"]
X = df[features]

# Feature engineering: Creating non-linear terms in the pipeline
if poly_degree:
    steps.append(("Poly", PolynomialFeatures(degree=poly_degree)))
Question 3: Evaluation Metrics and Model InterpretationQuestion: Three metrics are logged to MLflow (MAE, RMSE, R2). Explain what each measures and what the final scores of the Polynomial Degree 3 model indicate.Answer:MAE (Mean Absolute Error): Measures the average prediction error in megawatts. The best model was off by ~3.61 MW.RMSE (Root Mean Squared Error): Measures error but penalizes large "misses" more heavily.R2 (R-Squared): Provides a percentage grade of how much variance the model explains. A score of 0.815 indicates the model explained 81.5% of the power changes.The jump from an R2 of 0.72 (Baseline) to 0.81 (Poly 3) proves that the cubic wind speed feature is a significantly better representation of the turbine's physical output than a simple straight line.Python# Calculating and logging metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2", r2)
Question 4: Preventing Data Leakage via PipelinesQuestion: How does bundling the MinMaxScaler and the model into a single Pipeline prevent errors during the evaluation phase?Answer:Data leakage occurs if the model "peeks" at the test set. For scaling, the minimum and maximum values must only be learned from the training set.By using a Pipeline, the MinMaxScaler learns the scale from X_train during the .fit() call. When .predict() is called on X_test, the pipeline automatically uses those training-set parameters to transform the test data. This ensures the model treats the test data as truly "unseen" future data.Python# Bundling steps to prevent leakage
pipeline = Pipeline([
    ("Scaler", MinMaxScaler()),
    ("Model", LinearRegression())
])
pipeline.fit(X_train, y_train)
Question 5: Reproducibility and Deployment via MLflowQuestion: What are the advantages of using mlflow.sklearn.log_model instead of just saving a standard pickle file?Answer:A standard pickle file only saves the model logic. mlflow.sklearn.log_model saves the entire "bundle," including the pipeline, the Python environment dependencies (conda.yaml), and the data schema (signature).This ensures that another researcher or an automated deployment system can recreate the exact same environment and knows precisely what data format (columns and types) the model requires. The input_example provides a physical reference for the required data structure.Python# Saving the full environment and signature
signature = infer_signature(X_train, y_pred)
mlflow.sklearn.log_model(
    pipeline, 
    "model", 
    signature=signature, 
    input_example=X_train.iloc[:1]
)