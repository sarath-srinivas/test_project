import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
from custom_transformers import CombinedAttributesAdder


def load_data():
    strat_train_set = pd.read_csv('strat_train_set.csv')
    strat_test_set = pd.read_csv('strat_test_set.csv')
    return strat_train_set, strat_test_set

def score_model():
    with mlflow.start_run(run_name="Model Scoring",nested=True):
        strat_train_set, strat_test_set = load_data()

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        num_attribs = list(housing.drop("ocean_proximity", axis=1))
        cat_attribs = ["ocean_proximity"]

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

        # Load the trained model
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        # Test data
        housing_test = strat_test_set.drop("median_house_value", axis=1)
        housing_test_labels = strat_test_set["median_house_value"].copy()
        housing_test_prepared = full_pipeline.transform(housing_test)

        final_predictions = lin_reg.predict(housing_test_prepared)
        final_mse = mean_squared_error(housing_test_labels, final_predictions)
        final_rmse = np.sqrt(final_mse)

        mlflow.log_metric("final_rmse", final_rmse)

if __name__ == "__main__":
    score_model()
