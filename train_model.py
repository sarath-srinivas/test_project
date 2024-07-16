import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import joblib
import psutil


from custom_transformers import CombinedAttributesAdder

def log_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_memory = psutil.virtual_memory()
    mlflow.log_metric("cpu_percent", cpu_percent)
    mlflow.log_metric("memory_percent", virtual_memory.percent)
    mlflow.log_metric("memory_used", virtual_memory.used)
    mlflow.log_metric("memory_available", virtual_memory.available)
def load_data():
    strat_train_set = pd.read_csv('strat_train_set.csv')
    strat_test_set = pd.read_csv('strat_test_set.csv')
    return strat_train_set, strat_test_set

def train_model():
    with mlflow.start_run(run_name="Model Training",nested=True):
        log_system_metrics()

        strat_train_set, strat_test_set = load_data()

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        num_attribs = list(housing.drop("ocean_proximity", axis=1))
        cat_attribs = ["ocean_proximity"]
        mlflow.log_param("num_attribs", num_attribs)
        mlflow.log_param("cat_attribs", cat_attribs)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("imputer_strategy", "median")
        mlflow.log_param("scaling", "StandardScaler")
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        housing_prepared = full_pipeline.fit_transform(housing)
        from mlflow.models import infer_signature

        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        signature = infer_signature(housing_prepared, lin_reg.predict(housing_prepared))


        # Predictions
        housing_predictions = lin_reg.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)

        mlflow.log_metric("rmse", lin_rmse)
        pipeline_path = "full_pipeline.pkl"
        joblib.dump(full_pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)
        log_system_metrics()

        # Log additional artifacts
        # For example, logging feature importances if available
        # feature_importances = pd.Series(lin_reg.coef_, index=num_attribs + list(full_pipeline.named_transformers_['cat'].categories_[0]))
        # feature_importances.to_csv("feature_importances.csv")
        # mlflow.log_artifact("feature_importances.csv")

        # # Remove local copies if needed
        # os.remove(pipeline_path)
        # os.remove("feature_importances.csv")

if __name__ == "__main__":
    train_model()
